"""Research Agent for the autonomous research workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import LLMService
from src.models.schemas import (
    ResearchLimitation,
    ResearchState,
    SearchResult,
    SearchSelection,
    Source,
    SubQuestionPlan,
)
from src.tools.page_fetcher import PageFetchError, fetch_page_content
from src.tools.web_search import SearchError, web_search

import re

from groq import BadRequestError


PLAN_MIN = 4
PLAN_MAX = 5
PLAN_WORD_MIN = 5
SELECTION_SIZE = 3
CANDIDATE_SIZE = 5
PLAN_RETRY_LIMIT = 2
SELECTION_RETRY_LIMIT = 2


class ResearchPlanningError(RuntimeError):
    """Raised when a valid 4-5 sub-question plan cannot be produced."""


class SearchSelectionError(RuntimeError):
    """Raised when a valid top-3 search selection cannot be produced."""


def _normalize_sub_question(value: str) -> str:
    normalized = " ".join(value.lower().strip().split())
    return normalized.rstrip("?!.,;:")


def _validate_sub_questions(plan: SubQuestionPlan) -> tuple[bool, str]:
    questions = plan.sub_questions

    if not PLAN_MIN <= len(questions) <= PLAN_MAX:
        return False, f"expected {PLAN_MIN}-{PLAN_MAX} sub-questions, got {len(questions)}"

    normalized_questions: set[str] = set()

    for position, question in enumerate(questions, start=1):
        if not isinstance(question, str) or not question.strip():
            return False, f"sub-question {position} is empty or not a string"

        if len(question.split()) < PLAN_WORD_MIN:
            return (
                False,
                f"sub-question {position} has fewer than {PLAN_WORD_MIN} words",
            )

        normalized = _normalize_sub_question(question)
        if not normalized:
            return False, f"sub-question {position} is empty after normalization"

        if normalized in normalized_questions:
            return False, f"duplicate sub-question at position {position}"

        normalized_questions.add(normalized)

    return True, ""


def _plan_messages(topic: str, feedback: str | None = None) -> list:
    system = (
        "You are the planning component of an autonomous research agent. "
        "For the user's research topic, produce a focused research plan with "
        "exactly 4 or 5 distinct sub-questions. Cover the core scope, important "
        "dimensions, current evidence, limitations or risks, and another "
        "topic-specific angle where useful. Keep every sub-question specific "
        "enough to search independently. Do not answer the questions."
    )
    user = f"Research topic:\n{topic.strip()}"

    if feedback:
        user += (
            "\n\nThe previous structured output failed deterministic validation. "
            f"Fix this exact issue: {feedback}. Return a corrected plan only."
        )

    return [SystemMessage(content=system), HumanMessage(content=user)]


def generate_sub_questions(topic: str, llm_service: LLMService) -> list[str]:
    feedback: str | None = None

    for attempt in range(PLAN_RETRY_LIMIT + 1):
        plan = llm_service.invoke_structured(
            _plan_messages(topic, feedback),
            SubQuestionPlan,
        )

        valid, reason = _validate_sub_questions(plan)
        if valid:
            return plan.sub_questions

        if attempt == PLAN_RETRY_LIMIT:
            break

        feedback = reason

    raise ResearchPlanningError(
        "Unable to produce a valid 4-5 sub-question plan after "
        f"{PLAN_RETRY_LIMIT} validation retries."
    )


def _selection_messages(
    topic: str,
    sub_question: str,
    candidates: list[SearchResult],
    feedback: str | None = None,
) -> list:
    candidate_lines = "\n".join(
        f"{index}. {candidate.title}\n"
        f"URL: {candidate.url}\n"
        f"Snippet: {candidate.snippet}"
        for index, candidate in enumerate(candidates, start=1)
    )

    system = (
        "You are the source-selection component of an autonomous research "
        "agent. Select exactly 3 of the 5 supplied web-search candidates that "
        "are most directly relevant to the sub-question. Prefer candidates "
        "whose title and snippet indicate direct evidence. Return only integer "
        "indices from 1 through 5. Do not invent indices or explain the choice."
    )

    user = (
        f"Overall research topic:\n{topic.strip()}\n\n"
        f"Sub-question:\n{sub_question.strip()}\n\n"
        f"Search candidates:\n{candidate_lines}"
    )

    if feedback:
        user += (
            "\n\nThe previous selection failed deterministic validation. "
            f"Fix this exact issue: {feedback}. Return a corrected selection only."
        )

    return [SystemMessage(content=system), HumanMessage(content=user)]


def _extract_recovered_selection(
    exc: BadRequestError,
) -> list[int] | None:
    """Extract an explicit three-index selection from malformed Groq output."""
    body = getattr(exc, "body", None)

    if not isinstance(body, dict):
        return None

    error = body.get("error")
    if not isinstance(error, dict):
        return None

    if error.get("code") not in {"output_parse_failed", "json_validate_failed"}:
        return None

    failed_generation = error.get("failed_generation")
    if not isinstance(failed_generation, str) or not failed_generation.strip():
        return None

    text = failed_generation.strip()

    # Only accept an explicit list-like selection pattern.
    patterns = (
        r'"selected_indices"\s*:\s*\[\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])',
        r"(?i)\b(?:likely|select(?:ed)?|selection|choose|pick)\b"
        r".*?\b([1-5])\s*[, ]\s*([1-5])\s*[, ]\s*([1-5])\b",
        r"\[\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*\]",
    )

    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue

        indices = [int(match.group(i)) for i in range(1, 4)]

        selection = SearchSelection(selected_indices=indices)
        valid, _ = _validate_selection(selection)

        if valid:
            return indices

    return None


def _validate_selection(selection: SearchSelection) -> tuple[bool, str]:
    indices = selection.selected_indices

    if len(indices) != SELECTION_SIZE:
        return (
            False,
            f"expected exactly {SELECTION_SIZE} selected indices, got {len(indices)}",
        )

    if len(set(indices)) != SELECTION_SIZE:
        return False, "selected indices must be unique"

    invalid = [index for index in indices if not 1 <= index <= CANDIDATE_SIZE]
    if invalid:
        return False, f"selected indices must be between 1 and {CANDIDATE_SIZE}"

    return True, ""


def _select_search_candidates_with_recovery(
    user_topic: str,
    sub_question: str,
    candidates: list[SearchResult],
    llm_service: LLMService,
) -> tuple[list[SearchResult], bool]:
    if len(candidates) != CANDIDATE_SIZE:
        raise SearchSelectionError(
            f"Expected exactly {CANDIDATE_SIZE} candidates for LLM selection, "
            f"got {len(candidates)}."
        )

    feedback = None

    for attempt in range(SELECTION_RETRY_LIMIT + 1):
        try:
            selection = llm_service.invoke_structured(
                _selection_messages(
                    user_topic,
                    sub_question,
                    candidates,
                    feedback,
                ),
                SearchSelection,
            )
        except BadRequestError as exc:
            recovered = _extract_recovered_selection(exc)
            if recovered is not None:
                return [candidates[index - 1] for index in recovered], True
            raise

        valid, reason = _validate_selection(selection)

        if valid:
            return [
                candidates[index - 1]
                for index in selection.selected_indices
            ], False

        if attempt == SELECTION_RETRY_LIMIT:
            break

        feedback = reason

    raise SearchSelectionError(
        "Search candidate selection remained invalid after "
        f"{SELECTION_RETRY_LIMIT + 1} attempts."
    )


def select_search_candidates(
    user_topic: str,
    sub_question: str,
    candidates: list[SearchResult],
    llm_service: LLMService,
) -> list[SearchResult]:
    selected, _ = _select_search_candidates_with_recovery(
        user_topic,
        sub_question,
        candidates,
        llm_service,
    )
    return selected


def _add_source(
    *,
    source_by_url: dict[str, Source],
    ordered_sources: list[Source],
    candidate: SearchResult,
    content: str,
    sub_question: str,
    citation_counter: list[int],
) -> None:
    existing = source_by_url.get(candidate.url)

    if existing is not None:
        if sub_question not in existing.search_queries:
            existing.search_queries.append(sub_question)
        return

    citation_id = citation_counter[0]
    citation_counter[0] += 1

    source = Source(
        citation_id=citation_id,
        title=candidate.title,
        url=candidate.url,
        retrieved_at=datetime.now(timezone.utc),
        snippet=candidate.snippet,
        search_queries=[sub_question],
        content=content,
    )

    source_by_url[candidate.url] = source
    ordered_sources.append(source)


def _candidate_attempt_order(
    search_results: list[SearchResult],
    selected: list[SearchResult] | None,
) -> list[SearchResult]:
    """Return selected candidates first, then remaining original results."""
    if selected is None:
        return list(search_results)

    selected_urls = {item.url for item in selected}
    ordered = list(selected)
    ordered.extend(
        item for item in search_results if item.url not in selected_urls
    )
    return ordered


def _attempt_candidates(
    *,
    sub_question: str,
    candidates: Iterable[SearchResult],
    source_by_url: dict[str, Source],
    ordered_sources: list[Source],
    citation_counter: list[int],
    failed_urls: set[str] | None = None,
) -> ResearchLimitation:
    """Fetch candidate pages and return honest per-question accounting."""
    usable_urls: set[str] = set()
    attempted_urls: set[str] = set()
    failed_count = 0

    for candidate in candidates:
        if candidate.url in usable_urls:
            continue

        existing = source_by_url.get(candidate.url)
        if existing is not None:
            if sub_question not in existing.search_queries:
                existing.search_queries.append(sub_question)
            usable_urls.add(candidate.url)
            continue

        if candidate.url in attempted_urls:
            continue

        attempted_urls.add(candidate.url)

        try:
            content = fetch_page_content(candidate.url)
        except PageFetchError:
            failed_count += 1
            if failed_urls is not None:
                failed_urls.add(candidate.url)
            continue

        _add_source(
            source_by_url=source_by_url,
            ordered_sources=ordered_sources,
            candidate=candidate,
            content=content,
            sub_question=sub_question,
            citation_counter=citation_counter,
        )
        usable_urls.add(candidate.url)

    usable_count = len(usable_urls)

    description = (
        f"{usable_count} usable unique URL(s) contributed to this sub-question; "
        f"{failed_count} candidate page(s) actually attempted and found unusable."
    )

    if usable_count < SELECTION_SIZE:
        description += (
            " Fewer than three usable sources were available after page retrieval."
        )

    return ResearchLimitation(
        sub_question=sub_question,
        usable_source_count=usable_count,
        failed_candidate_count=failed_count,
        description=description,
    )


def run_research(
    state: ResearchState,
    llm_service: LLMService,
    on_selection_recovery: Callable[[str], None] | None = None,
) -> ResearchState:
    sub_questions = generate_sub_questions(state.user_topic, llm_service)

    source_by_url = {
        source.url: source.model_copy(deep=True)
        for source in state.sources
    }
    ordered_sources = list(source_by_url.values())

    next_citation_id = (
        max((source.citation_id for source in ordered_sources), default=0) + 1
    )
    citation_counter = [next_citation_id]
    limitations: list[ResearchLimitation] = []

    for sub_question in sub_questions:
        try:
            search_results = web_search(
                sub_question,
                max_results=CANDIDATE_SIZE,
            )
        except SearchError as exc:
            limitations.append(
                ResearchLimitation(
                    sub_question=sub_question,
                    usable_source_count=0,
                    failed_candidate_count=0,
                    description=(
                        "Web search failed for this sub-question; "
                        f"research could not retrieve candidates ({exc})."
                    ),
                )
            )
            continue

        if not search_results:
            limitations.append(
                ResearchLimitation(
                    sub_question=sub_question,
                    usable_source_count=0,
                    failed_candidate_count=0,
                    description=(
                        "Web search returned no candidates for this sub-question."
                    ),
                )
            )
            continue

        selected: list[SearchResult] | None = None
        selection_recovered_locally = False

        if len(search_results) == CANDIDATE_SIZE:
            selected, selection_recovered_locally = (
                _select_search_candidates_with_recovery(
                    state.user_topic,
                    sub_question,
                    search_results,
                    llm_service,
                )
            )

        if selection_recovered_locally and on_selection_recovery is not None:
            on_selection_recovery(sub_question)

        attempt_order = _candidate_attempt_order(search_results, selected)
        limitation = _attempt_candidates(
            sub_question=sub_question,
            candidates=attempt_order,
            source_by_url=source_by_url,
            ordered_sources=ordered_sources,
            citation_counter=citation_counter,
        )

        if len(search_results) < CANDIDATE_SIZE:
            limitation.description = (
                f"Search returned {len(search_results)} candidate result(s), "
                f"fewer than the expected {CANDIDATE_SIZE}; used all available "
                "candidates directly without LLM selection. "
                + limitation.description
            )

        limitations.append(limitation)

    return state.model_copy(
        update={
            "sub_questions": sub_questions,
            "sources": ordered_sources,
            "research_limitations": limitations,
        }
    )


def research_node(state: ResearchState) -> dict:
    """LangGraph-compatible Research Agent node."""
    from src.core.llm import get_llm_service

    updated_state = run_research(state, get_llm_service())

    return {
        "sub_questions": updated_state.sub_questions,
        "sources": updated_state.sources,
        "research_limitations": updated_state.research_limitations,
    }
