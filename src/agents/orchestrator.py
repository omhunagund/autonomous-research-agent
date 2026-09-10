"""Workflow orchestration and Critic-driven correction control."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from groq import BadRequestError

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.analysis_agent import analyze_research
from src.agents.critic_agent import critique_report
from src.agents.research_agent import (
    CANDIDATE_SIZE,
    SELECTION_SIZE,
    _add_source,
    _attempt_candidates,
    _candidate_attempt_order,
    select_search_candidates,
)
from src.agents.writing_agent import write_report
from src.core.llm import LLMService
from src.models.schemas import (
    CorrectionQueryPlan,
    Critique,
    CritiqueCheck,
    FinalReport,
    QualityLevel,
    ResearchLimitation,
    ResearchState,
    SearchResult,
    Source,
)
from src.tools.web_search import SearchError, web_search

MAX_CORRECTION_CYCLES = 2
CORRECTION_QUERY_RETRY_LIMIT = 2
CORRECTION_QUERY_MIN = 1
CORRECTION_QUERY_MAX = 3

ISSUE_CATEGORY_RE = re.compile(r"^\[(RESEARCH|WRITING|ANALYSIS|INFO)\]\s+(.+?)\s*$")
QUOTED_SUBQUESTION_RE = re.compile(r'"([^"]+)"')


class OrchestrationError(RuntimeError):
    """Raised for unrecoverable orchestration failures."""


class CorrectionIssueMappingError(OrchestrationError):
    """Raised when a Research issue cannot be mapped to a current sub-question."""


class CorrectionQueryValidationError(OrchestrationError):
    """Raised when targeted correction queries remain structurally invalid."""


@dataclass(frozen=True)
class CorrectionGroup:
    sub_question: str
    issues: tuple[str, ...]


def _normalize_sub_question(value: str) -> str:
    normalized = " ".join(value.lower().strip().split())
    return normalized.rstrip("?!.,;:")


def _parse_issue(issue: str) -> tuple[str, str] | None:
    match = ISSUE_CATEGORY_RE.match(issue.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def _extract_subquestion_from_issue(issue: str) -> str:
    parsed = _parse_issue(issue)
    if parsed is None:
        raise CorrectionIssueMappingError(
            f"Malformed Critic issue cannot be used for correction: {issue!r}"
        )

    category, body = parsed
    if category != "RESEARCH":
        raise CorrectionIssueMappingError(
            f"Only [RESEARCH] issues can be mapped to Research correction: {issue!r}"
        )

    quoted = QUOTED_SUBQUESTION_RE.search(body)
    if quoted is None or not quoted.group(1).strip():
        raise CorrectionIssueMappingError(
            "[RESEARCH] correction issue must contain the exact current "
            f"sub-question in quotes: {issue!r}"
        )

    return quoted.group(1).strip()


def group_research_issues(
    issues: Iterable[str],
    current_sub_questions: list[str],
) -> list[CorrectionGroup]:
    """Group Research issues by exact normalized current sub-question text."""
    by_normalized = {
        _normalize_sub_question(question): question
        for question in current_sub_questions
    }
    grouped: dict[str, list[str]] = defaultdict(list)

    for issue in issues:
        quoted = _extract_subquestion_from_issue(issue)
        normalized = _normalize_sub_question(quoted)
        current = by_normalized.get(normalized)
        if current is None:
            raise CorrectionIssueMappingError(
                "Critic [RESEARCH] issue references a sub-question that is not "
                f"present in current state: {quoted!r}"
            )
        grouped[normalized].append(issue)

    return [
        CorrectionGroup(
            sub_question=by_normalized[key],
            issues=tuple(grouped[key]),
        )
        for key in grouped
    ]


def _correction_query_messages(
    topic: str,
    sub_question: str,
    issues: tuple[str, ...],
    feedback: str | None = None,
) -> list:
    issue_block = "\n".join(f"- {issue}" for issue in issues)
    system = (
        "You are the targeted search-planning component of an autonomous "
        "research agent. Produce 1 to 3 complementary web-search queries "
        "that directly address the cited research deficiency for the exact "
        "sub-question. The queries must repair missing, weak, or outdated "
        "evidence without changing the approved sub-question itself. Avoid "
        "duplicates and make each query materially complementary. "
        "Return exactly one structured object with a single field named "
        "'queries', whose value is an array of 1 to 3 query strings. "
        "Do not return prose, explanations, headings, markdown, or code fences. "
        'The required shape is: {"queries": ["query 1", "query 2"]}.'
    )
    user = (
        f"Research topic:\n{topic.strip()}\n\n"
        f"Current sub-question:\n{sub_question.strip()}\n\n"
        f"Latest Critic research issues:\n{issue_block}"
    )
    if feedback:
        user += (
            "\n\nThe previous structured correction-query plan failed deterministic "
            f"validation. Fix this exact issue: {feedback}. Return corrected queries only."
        )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _validate_correction_queries(plan: CorrectionQueryPlan) -> tuple[bool, str]:
    if not CORRECTION_QUERY_MIN <= len(plan.queries) <= CORRECTION_QUERY_MAX:
        return False, (
            f"expected {CORRECTION_QUERY_MIN}-{CORRECTION_QUERY_MAX} correction queries, "
            f"got {len(plan.queries)}"
        )

    normalized: set[str] = set()
    for index, query in enumerate(plan.queries, start=1):
        if not isinstance(query, str) or not query.strip():
            return False, f"correction query {index} is empty or not a string"
        key = " ".join(query.lower().strip().split())
        if key in normalized:
            return False, f"duplicate correction query at position {index}"
        normalized.add(key)

    return True, ""


def _extract_recovered_correction_queries(
    exc: BadRequestError,
) -> list[str] | None:
    """Extract high-confidence queries from Groq's malformed structured output."""
    body = getattr(exc, "body", None)

    if not isinstance(body, dict):
        return None

    error = body.get("error")
    if not isinstance(error, dict):
        return None

    if error.get("code") != "output_parse_failed":
        return None

    failed_generation = error.get("failed_generation")
    if not isinstance(failed_generation, str) or not failed_generation.strip():
        return None

    quoted = re.findall(r'"([^"]+)"', failed_generation)

    if quoted:
        candidates = quoted
    elif "\n" in failed_generation:
        lines = [line.strip() for line in failed_generation.splitlines() if line.strip()]

        bullet_pattern = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
        marked = [
            match.group(1).strip()
            for line in lines
            if (match := bullet_pattern.match(line))
        ]

        if marked:
            candidates = marked
        else:
            candidates = lines
    else:
        return None

    normalized_candidates: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        query = " ".join(candidate.strip().split())
        if not query:
            continue

        key = query.lower()
        if key in seen:
            continue

        seen.add(key)
        normalized_candidates.append(query)

    return normalized_candidates[:CORRECTION_QUERY_MAX] or None


def generate_correction_queries(
    topic: str,
    sub_question: str,
    issues: tuple[str, ...],
    llm_service: LLMService,
) -> tuple[list[str], bool]:
    feedback: str | None = None

    for attempt in range(CORRECTION_QUERY_RETRY_LIMIT + 1):
        try:
            plan = llm_service.invoke_structured(
                _correction_query_messages(
                    topic,
                    sub_question,
                    issues,
                    feedback,
                ),
                CorrectionQueryPlan,
            )
        except BadRequestError as exc:
            recovered = _extract_recovered_correction_queries(exc)

            if recovered is None:
                raise

            recovered_plan = CorrectionQueryPlan(queries=recovered)
            valid, reason = _validate_correction_queries(recovered_plan)

            if valid:
                return recovered_plan.queries, True

            if attempt == CORRECTION_QUERY_RETRY_LIMIT:
                raise CorrectionQueryValidationError(
                    "Locally recovered correction queries failed validation: "
                    f"{reason}"
                ) from exc

            feedback = reason
            continue

        valid, reason = _validate_correction_queries(plan)

        if valid:
            return plan.queries, False

        if attempt == CORRECTION_QUERY_RETRY_LIMIT:
            break

        feedback = reason

    raise CorrectionQueryValidationError(
        "Unable to produce a valid targeted correction-query plan after "
        f"{CORRECTION_QUERY_RETRY_LIMIT} validation retries."
    )


def _attempt_correction_candidates(
    *,
    topic: str,
    sub_question: str,
    search_query: str,
    candidates: list[SearchResult],
    source_by_url: dict[str, Source],
    ordered_sources: list[Source],
    citation_counter: list[int],
    failed_urls: set[str],
    llm_service: LLMService,
) -> None:
    """Apply the identical candidate-selection contract to one correction query."""
    if len(candidates) == CANDIDATE_SIZE:
        selected = select_search_candidates(
            topic,
            sub_question,
            candidates,
            llm_service,
        )
    else:
        selected = None

    attempt_order = _candidate_attempt_order(candidates, selected)
    attempted_this_query: list[SearchResult] = []
    for candidate in attempt_order:
        if candidate.url in failed_urls:
            continue
        attempted_this_query.append(candidate)

    # _attempt_candidates calls the module-level fetcher and already performs
    # global URL deduplication.  We add a local failed-URL guard here so the
    # same unusable candidate is never retried across correction queries.
    _attempt_candidates(
        sub_question=sub_question,
        candidates=[c for c in attempted_this_query if c.url not in failed_urls],
        source_by_url=source_by_url,
        ordered_sources=ordered_sources,
        citation_counter=citation_counter,
        failed_urls=failed_urls,
    )


def run_research_correction(
    state: ResearchState,
    research_issues: list[str],
    llm_service: LLMService,
) -> tuple[ResearchState, bool]:
    """Execute targeted Research correction while preserving current state."""
    groups = group_research_issues(research_issues, state.sub_questions)
    if not groups:
        raise CorrectionIssueMappingError(
            "Research correction requires at least one issue"
        )

    source_by_url = {
        source.url: source.model_copy(deep=True) for source in state.sources
    }
    ordered_sources = list(source_by_url.values())
    citation_counter = [
        max((source.citation_id for source in ordered_sources), default=0) + 1
    ]
    failed_urls: set[str] = set()
    fresh_failed_by_subquestion: dict[str, set[str]] = defaultdict(set)
    search_failed_by_subquestion: set[str] = set()
    new_urls_by_subquestion: dict[str, set[str]] = defaultdict(set)
    recovered_query_generation = False

    for group in groups:
        queries, recovered = generate_correction_queries(
            state.user_topic,
            group.sub_question,
            group.issues,
            llm_service,
        )

        if recovered:
            recovered_query_generation = True

        for query in queries:
            try:
                candidates = web_search(query, max_results=CANDIDATE_SIZE)
            except SearchError:
                search_failed_by_subquestion.add(group.sub_question)
                continue

            if not candidates:
                continue

            before_urls = set(source_by_url)
            failed_before = set(failed_urls)

            _attempt_correction_candidates(
                topic=state.user_topic,
                sub_question=group.sub_question,
                search_query=query,
                candidates=candidates,
                source_by_url=source_by_url,
                ordered_sources=ordered_sources,
                citation_counter=citation_counter,
                failed_urls=failed_urls,
                llm_service=llm_service,
            )

            after_urls = set(source_by_url)

            fresh_failed_by_subquestion[group.sub_question].update(
                failed_urls - failed_before
            )
            new_urls_by_subquestion[group.sub_question].update(
                after_urls - before_urls
            )

    limitations_by_subquestion = {
        _normalize_sub_question(item.sub_question): item
        for item in state.research_limitations
    }

    for group in groups:
        attributed_existing = {
            source.url
            for source in ordered_sources
            if group.sub_question in source.search_queries
        }

        usable_urls = (
            attributed_existing
            | new_urls_by_subquestion[group.sub_question]
        )

        failed_count = len(
            fresh_failed_by_subquestion[group.sub_question]
        )
        usable_count = len(usable_urls)

        normalized = _normalize_sub_question(group.sub_question)

        if usable_count >= SELECTION_SIZE:
            limitations_by_subquestion.pop(normalized, None)
            continue

        description = (
            f"Current correction accounting: {usable_count} usable unique URL(s) "
            f"for this sub-question and {failed_count} unique candidate page(s) "
            "actually attempted and found unusable."
        )

        if group.sub_question in search_failed_by_subquestion:
            description += " At least one targeted web search failed."

        description += " Fewer than three usable sources are currently available."

        limitations_by_subquestion[normalized] = ResearchLimitation(
            sub_question=group.sub_question,
            usable_source_count=usable_count,
            failed_candidate_count=failed_count,
            description=description,
        )

    updated_state = state.model_copy(
        update={
            "sources": ordered_sources,
            "research_limitations": list(
                limitations_by_subquestion.values()
            ),
        }
    )

    return updated_state, recovered_query_generation


def _issue_categories(issues: Iterable[str]) -> set[str]:
    categories: set[str] = set()
    for issue in issues:
        parsed = _parse_issue(issue)
        if parsed is not None:
            categories.add(parsed[0])
    return categories


def _set_final_quality(state: ResearchState, quality: QualityLevel) -> ResearchState:
    if state.draft is None:
        raise OrchestrationError("Cannot finalize without a provisional draft")
    report = state.draft.model_copy(update={"quality": quality})
    return state.model_copy(update={"draft": report, "final_report": report})


def _finalize_analysis_only(state: ResearchState, critique: Critique) -> ResearchState:
    issues = [issue for issue in critique.issues if issue.startswith("[ANALYSIS]")]
    if state.draft is None:
        raise OrchestrationError("Cannot finalize without a provisional draft")
    report = state.draft.model_copy(
        update={
            "quality": QualityLevel.LOW,
            "analysis_issues": issues,
            "unresolved_issues": [],
        }
    )
    return state.model_copy(
        update={
            "draft": report,
            "final_report": report,
            "revision_target": "analysis",
        }
    )


def _finalize_unresolved(state: ResearchState, critique: Critique) -> ResearchState:
    unresolved = [
        issue
        for issue in critique.issues
        if issue.startswith("[RESEARCH]") or issue.startswith("[WRITING]")
    ]
    if state.draft is None:
        raise OrchestrationError("Cannot finalize without a provisional draft")
    report = state.draft.model_copy(
        update={
            "quality": QualityLevel.LOW,
            "analysis_issues": [],
            "unresolved_issues": unresolved,
        }
    )
    return state.model_copy(
        update={
            "draft": report,
            "final_report": report,
            "revision_target": "none",
        }
    )


def run_workflow(
    state: ResearchState,
    llm_service: LLMService,
) -> ResearchState:
    """Run Research -> Analysis -> Writing -> Critic with bounded correction."""
    from src.agents.research_agent import run_research

    current = run_research(state, llm_service)
    analysis = analyze_research(current, llm_service)
    current = current.model_copy(
        update={
            "findings": analysis.findings,
            "conflicts": analysis.conflicts,
            "gaps": analysis.gaps,
        }
    )
    current = current.model_copy(update={"draft": write_report(current, llm_service)})

    initial_failure = False

    while True:
        critique, target = critique_report(current, llm_service)
        current = current.model_copy(update={"critique": critique, "revision_target": target})

        if critique.verdict == CritiqueCheck.PASS:
            quality = QualityLevel.HIGH if not initial_failure else QualityLevel.MEDIUM
            current = _set_final_quality(current, quality)
            return current.model_copy(update={"revision_target": "none"})

        categories = _issue_categories(critique.issues)
        if categories and categories <= {"ANALYSIS", "INFO"} and "ANALYSIS" in categories:
            return _finalize_analysis_only(current, critique)

        if target not in {"research", "writing"}:
            return _finalize_unresolved(current, critique)

        if current.retry_count >= MAX_CORRECTION_CYCLES:
            return _finalize_unresolved(current, critique)

        initial_failure = True
        next_count = current.retry_count + 1
        current = current.model_copy(update={"retry_count": next_count})

        if target == "research":
            research_issues = [issue for issue in critique.issues if issue.startswith("[RESEARCH]")]
            current, _ = run_research_correction(
                current,
                research_issues,
                llm_service,
            )
            analysis = analyze_research(current, llm_service)
            current = current.model_copy(
                update={
                    "findings": analysis.findings,
                    "conflicts": analysis.conflicts,
                    "gaps": analysis.gaps,
                    "draft": None,
                    "critique": None,
                    "revision_target": None,
                }
            )
            current = current.model_copy(update={"draft": write_report(current, llm_service)})
        else:
            writing_issues = [issue for issue in critique.issues if issue.startswith("[WRITING]")]
            current = current.model_copy(update={"revision_target": "writing"})
            current = current.model_copy(update={"draft": write_report(current, llm_service, correction_issues=writing_issues)})
            current = current.model_copy(update={"revision_target": "writing"})


def orchestrator_node(state: ResearchState) -> ResearchState:
    """Convenience entry point using the configured shared LLM service."""
    from src.core.llm import get_llm_service

    return run_workflow(state, get_llm_service())
