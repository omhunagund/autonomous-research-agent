"""Critic Agent for semantic review of a provisional research report.

The Critic:
- reviews only a structurally valid provisional Writer artifact;
- evaluates Faithfulness, Coverage, Recency, and Balance independently;
- returns PASS/FAIL component assessments;
- provides actionable categorized issues;
- does not repair the report;
- derives the aggregate verdict and correction target deterministically.

Routing issue prefixes:
    [RESEARCH] -> evidence/retrieval deficiency
    [WRITING]  -> Writer presentation/fidelity deficiency
    [ANALYSIS] -> upstream analytical deficiency
    [INFO]     -> non-material observation
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.evidence_compaction import compact_sources_for_prompt

from src.core.llm import LLMService
from src.models.schemas import (
    Critique,
    CritiqueAssessment,
    CritiqueCheck,
    FinalReport,
    ResearchState,
)


CRITIC_RETRY_LIMIT = 2
ISSUE_PREFIXES = ("[RESEARCH]", "[WRITING]", "[ANALYSIS]", "[INFO]")
ISSUE_PREFIX_PATTERN = re.compile(r"^\[(RESEARCH|WRITING|ANALYSIS|INFO)\]\s+(.+)$")
REVISION_TARGETS = {"research", "writing", "analysis", "none"}


class CriticInputValidationError(RuntimeError):
    """Raised when the provisional Writer artifact is not reviewable."""


class CritiqueValidationError(RuntimeError):
    """Raised when the Critic response violates its deterministic contract."""


def _parse_issue(issue: str) -> tuple[str, str] | None:
    match = ISSUE_PREFIX_PATTERN.match(issue.strip())
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _issue_categories(issues: Iterable[str]) -> set[str]:
    categories: set[str] = set()
    for issue in issues:
        parsed = _parse_issue(issue)
        if parsed is not None:
            categories.add(parsed[0])
    return categories


def _normalize_sub_question(value: str) -> str:
    return " ".join(value.lower().strip().split()).rstrip("?!.,;:")


def _validate_research_issue_subquestion(
    issue: str,
    current_sub_questions: list[str],
) -> tuple[bool, str]:
    parsed = _parse_issue(issue)
    if parsed is None:
        return False, "issue format is invalid"

    category, body = parsed
    if category != "RESEARCH":
        return True, ""

    quoted = re.search(r'"([^"]+)"', body)
    if quoted is None or not quoted.group(1).strip():
        return False, (
            "[RESEARCH] issue must contain exactly one current "
            "sub-question in quotes"
        )

    quoted_normalized = _normalize_sub_question(quoted.group(1))
    matches = [
        question
        for question in current_sub_questions
        if _normalize_sub_question(question) == quoted_normalized
    ]

    if len(matches) != 1:
        return False, (
            "[RESEARCH] issue must quote exactly one current approved "
            "sub-question verbatim; do not combine multiple sub-questions "
            "into one issue"
        )

    return True, ""


def _validate_reviewable_draft(state: ResearchState) -> None:
    """Defensively validate the provisional Writer artifact before review."""
    draft = state.draft
    if draft is None:
        raise CriticInputValidationError(
            "No provisional Writer draft is present for Critic review."
        )

    source_ids = {source.citation_id for source in state.sources}

    if draft.topic != state.user_topic:
        raise CriticInputValidationError(
            "Writer draft topic does not match the current research topic."
        )

    if len(draft.key_findings) != len(state.findings):
        raise CriticInputValidationError(
            "Writer draft findings do not match the Analysis finding count."
        )

    if draft.key_findings != state.findings:
        raise CriticInputValidationError(
            "Writer draft Finding objects do not preserve the Analysis baseline."
        )

    if draft.gaps != state.gaps:
        raise CriticInputValidationError(
            "Writer draft Gap objects do not preserve the Analysis baseline."
        )

    if draft.conflicts != state.conflicts:
        raise CriticInputValidationError(
            "Writer draft Conflict objects do not preserve the Analysis baseline."
        )

    if len(draft.gap_explanations) != len(state.gaps):
        raise CriticInputValidationError(
            "Writer draft gap explanations do not match the Analysis gap count."
        )

    if len(draft.conflict_explanations) != len(state.conflicts):
        raise CriticInputValidationError(
            "Writer draft conflict explanations do not match the Analysis conflict count."
        )

    for finding in draft.key_findings:
        if not finding.supporting_sources:
            raise CriticInputValidationError(
                "A Finding has no supporting sources; the Writer artifact is invalid."
            )
        if set(finding.supporting_sources) - source_ids:
            raise CriticInputValidationError(
                "A Finding references an unknown source citation ID."
            )

    reference_ids = [source.citation_id for source in draft.references]
    if reference_ids != sorted(reference_ids):
        raise CriticInputValidationError(
            "Draft references are not ordered by citation_id."
        )

    if len(reference_ids) != len(set(reference_ids)):
        raise CriticInputValidationError(
            "Draft references contain duplicate citation IDs."
        )

    if set(reference_ids) - source_ids:
        raise CriticInputValidationError(
            "Draft references contain unknown citation IDs."
        )


def _source_context(state: ResearchState) -> str:
    if not state.sources:
        return "No current retrieved sources."

    compacted_sources = compact_sources_for_prompt(
        state.sources,
        state.sub_questions,
    )

    if not compacted_sources:
        return "No current retrieved source content is available within the prompt budget."

    return "\n\n---\n\n".join(
        (
            f"Source [{source.citation_id}]\n"
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Retrieved at: {source.retrieved_at.isoformat()}\n"
            f"Search queries: {', '.join(source.search_queries)}\n"
            f"Snippet: {source.snippet}\n"
            f"Content:\n{source.content}"
        )
        for source in compacted_sources
    )


def _analysis_context(state: ResearchState) -> str:
    findings = "\n".join(
        (
            f"{index}. claim={finding.claim!r}; "
            f"supporting_sources={finding.supporting_sources}; "
            f"confidence={finding.confidence.value}"
        )
        for index, finding in enumerate(state.findings, start=1)
    ) or "No findings."

    gaps = "\n".join(
        (
            f"{index}. type={gap.type.value}; "
            f"description={gap.description!r}; "
            f"related_sub_question={gap.related_sub_question!r}; "
            f"related_claim={gap.related_claim!r}"
        )
        for index, gap in enumerate(state.gaps, start=1)
    ) or "No gaps."

    conflicts = "\n".join(
        (
            f"{index}. description={conflict.description!r}; "
            f"related_sources={conflict.related_sources}"
        )
        for index, conflict in enumerate(state.conflicts, start=1)
    ) or "No conflicts."

    return (
        f"Findings:\n{findings}\n\n"
        f"Gaps:\n{gaps}\n\n"
        f"Conflicts:\n{conflicts}"
    )


def _limitations_context(state: ResearchState) -> str:
    if not state.research_limitations:
        return "No research limitations."

    return "\n".join(
        (
            f"- {item.sub_question}: usable={item.usable_source_count}, "
            f"failed_attempts={item.failed_candidate_count}; {item.description}"
        )
        for item in state.research_limitations
    )


def _draft_context(draft: FinalReport) -> str:
    findings = "\n".join(
        (
            f"{index}. {finding.claim} "
            f"(confidence={finding.confidence.value}; "
            f"supporting_sources={finding.supporting_sources})"
        )
        for index, finding in enumerate(draft.key_findings, start=1)
    ) or "No key findings."

    evidence = "\n".join(
        (
            f"- {entry.text} "
            f"(citation_ids={entry.citation_ids}; "
            f"related_finding_indices={entry.related_finding_indices})"
        )
        for entry in draft.supporting_evidence
    ) or "No supporting evidence entries."

    gaps = "\n".join(
        (
            f"{index}. {gap.type.value}: {gap.description}\n"
            f"   explanation: {draft.gap_explanations[index - 1]}"
        )
        for index, gap in enumerate(draft.gaps, start=1)
    ) or "No gaps."

    conflicts = "\n".join(
        (
            f"{index}. {conflict.description}\n"
            f"   related_sources={conflict.related_sources}\n"
            f"   explanation: {draft.conflict_explanations[index - 1]}"
        )
        for index, conflict in enumerate(draft.conflicts, start=1)
    ) or "No conflicts."

    references = "\n".join(
        f"- [{source.citation_id}] {source.title} ({source.url})"
        for source in draft.references
    ) or "No references."

    return (
        f"Executive summary:\n{draft.executive_summary}\n\n"
        f"Key findings:\n{findings}\n\n"
        f"Supporting evidence:\n{evidence}\n\n"
        f"Gaps:\n{gaps}\n\n"
        f"Conflicts:\n{conflicts}\n\n"
        f"References:\n{references}"
    )


def _critic_messages(
    state: ResearchState,
    feedback: str | None = None,
) -> list:
    system = (
        "You are the Critic Agent in an autonomous research and report "
        "system. Review the supplied structurally valid provisional report "
        "against the current Analysis, fresh Sources, sub-questions, and "
        "research limitations. Evaluate four checks independently and always "
        "return all four: Faithfulness, Coverage, Recency, Balance. Use only "
        "PASS or FAIL for each. Do not repair the report. Do not reassign "
        "Finding confidence. Do not search the web. Do not use memory as "
        "evidence.\n\n"
        "FAITHFULNESS: determine whether the written report faithfully "
        "represents the established Analysis and retrieved evidence. A "
        "Writer-selected subset of a Finding's supporting sources is allowed "
        "when that subset faithfully supports the written claim and does not "
        "omit a material qualification. Writer-introduced overclaiming, "
        "distortion, evidence expansion, suppressed gaps/conflicts, or "
        "misleading summaries are Writing failures.\n\n"
        "COVERAGE: determine whether every approved sub-question receives "
        "meaningful treatment through findings, evidence, an honest gap, or "
        "a relevant conflict. Mere keyword mention is not coverage. A "
        "sub-question can be covered honestly by an explicit research/evidence "
        "gap. Diagnose whether a failure is caused by missing retrieved "
        "evidence ([RESEARCH]), Writer omission ([WRITING]), or an upstream "
        "Analysis deficiency ([ANALYSIS]).\n\n"
        "RECENCY: evaluate claim-by-claim and contextually. Historical or "
        "stable facts can appropriately use older sources. Current-state "
        "claims on fast-changing topics require reasonably current retrieved "
        "evidence or clear historical qualification. Never use 'newest wins' "
        "and never re-search. Missing current evidence is [RESEARCH]; "
        "misrepresenting dated evidence as current is [WRITING].\n\n"
        "BALANCE: determine whether presentation fairly reflects material "
        "conflicts, uncertainty, gaps, and contrary evidence. Balance is "
        "proportionate to evidence, not equal word count or false symmetry. "
        "A Writer framing problem is [WRITING]. If the Analysis baseline itself "
        "omitted a material perspective/conflict, use [ANALYSIS].\n\n"
        "ISSUES: every material failure must have a specific issue beginning "
        "with exactly one of [RESEARCH], [WRITING], [ANALYSIS], [INFO]. "
        "Faithfulness FAIL requires [WRITING]. Coverage FAIL requires at "
        "least one of [RESEARCH]/[WRITING]/[ANALYSIS]. Recency FAIL requires "
        "[RESEARCH] or [WRITING]. Balance FAIL requires [WRITING] or "
        "[ANALYSIS]. [INFO] is for non-material observations and cannot be "
        "the only explanation for a failed check. "
        "For every [RESEARCH] issue, quote exactly one current approved "
        "sub-question from the list above verbatim and identify the specific "
        "evidence deficiency for that sub-question. Do not combine multiple "
        "sub-questions into one [RESEARCH] issue. Create separate [RESEARCH] "
        "issues when multiple sub-questions have separate evidence "
        "deficiencies. [WRITING], [ANALYSIS], and [INFO] issues do not require "
        "a quoted sub-question unless useful. One underlying defect may "
        "legitimately produce distinct issues for multiple checks.\n\n"
        "Do not return an aggregate verdict field. The application will derive "
        "the final verdict deterministically from the four component checks."
    )

    user = (
        f"Research topic:\n{state.user_topic}\n\n"
        f"Approved sub-questions:\n"
        + "\n".join(f"- {question}" for question in state.sub_questions)
        + "\n\n"
        f"Analysis baseline:\n{_analysis_context(state)}\n\n"
        f"Research limitations:\n{_limitations_context(state)}\n\n"
        f"Provisional report:\n{_draft_context(state.draft)}\n\n"
        f"Current retrieved sources:\n{_source_context(state)}"
    )

    if feedback:
        user += (
            "\n\nThe previous Critic response failed deterministic validation. "
            f"Correct this exact issue: {feedback}. Re-evaluate the four checks "
            "independently and return the complete corrected Critique."
        )

    return [SystemMessage(content=system), HumanMessage(content=user)]


def _required_categories_for_check(
    check_name: str,
) -> set[str]:
    return {
        "faithfulness": {"WRITING"},
        "coverage": {"RESEARCH", "WRITING", "ANALYSIS"},
        "recency": {"RESEARCH", "WRITING"},
        "balance": {"WRITING", "ANALYSIS"},
    }[check_name]


def _validate_critique(
    critique: Critique,
    current_sub_questions: list[str],
) -> tuple[bool, str]:
    if critique.verdict not in {CritiqueCheck.PASS, CritiqueCheck.FAIL}:
        return False, "verdict must be PASS or FAIL"

    expected_verdict = (
        CritiqueCheck.PASS
        if all(
            check == CritiqueCheck.PASS
            for check in (
                critique.faithfulness,
                critique.coverage,
                critique.recency,
                critique.balance,
            )
        )
        else CritiqueCheck.FAIL
    )

    if critique.verdict != expected_verdict:
        return False, (
            "verdict is inconsistent with the four component checks; "
            "derive PASS only when all four checks PASS"
        )

    if not isinstance(critique.issues, list):
        return False, "issues must be a list"

    for index, issue in enumerate(critique.issues, start=1):
        if not isinstance(issue, str) or not issue.strip():
            return False, f"issue {index} is empty"

        parsed = _parse_issue(issue)
        if parsed is None:
            return False, (
                f"issue {index} must begin with exactly one of "
                f"{', '.join(ISSUE_PREFIXES)} followed by explanatory text"
            )

        research_valid, research_reason = _validate_research_issue_subquestion(
            issue,
            current_sub_questions,
        )
        if not research_valid:
            return False, f"issue {index}: {research_reason}"

    failed_checks = {
        "faithfulness": critique.faithfulness,
        "coverage": critique.coverage,
        "recency": critique.recency,
        "balance": critique.balance,
    }

    categories = _issue_categories(critique.issues)

    for check_name, result in failed_checks.items():
        if result != CritiqueCheck.FAIL:
            continue

        allowed = _required_categories_for_check(check_name)
        if not categories & allowed:
            return False, (
                f"{check_name} is FAIL but no compatible non-INFO issue "
                f"was provided; expected one of {sorted(allowed)}"
            )

    # A non-INFO issue must explain every failure; INFO alone cannot do so.
    if any(result == CritiqueCheck.FAIL for result in failed_checks.values()):
        if categories <= {"INFO"}:
            return False, "INFO issues cannot be the sole explanation for a failed check"

    # A PASS verdict may have informational observations.
    return True, ""


def derive_revision_target(critique: Critique) -> str:
    """Derive the initial correction target from validated issue categories."""
    if critique.verdict == CritiqueCheck.PASS:
        return "none"

    categories = _issue_categories(critique.issues)

    if "RESEARCH" in categories:
        return "research"

    if "WRITING" in categories:
        return "writing"

    if "ANALYSIS" in categories:
        return "analysis"

    return "none"


def critique_report(
    state: ResearchState,
    llm_service: LLMService,
) -> tuple[Critique, str]:
    """Review a provisional report and return validated critique + target."""
    _validate_reviewable_draft(state)

    feedback: str | None = None

    for attempt in range(CRITIC_RETRY_LIMIT + 1):
        assessment = llm_service.invoke_structured(
            _critic_messages(state, feedback),
            CritiqueAssessment,
        )

        verdict = (
            CritiqueCheck.PASS
            if all(
                check == CritiqueCheck.PASS
                for check in (
                    assessment.faithfulness,
                    assessment.coverage,
                    assessment.recency,
                    assessment.balance,
                )
            )
            else CritiqueCheck.FAIL
        )

        critique = Critique(
            faithfulness=assessment.faithfulness,
            coverage=assessment.coverage,
            recency=assessment.recency,
            balance=assessment.balance,
            verdict=verdict,
            issues=assessment.issues,
        )

        valid, reason = _validate_critique(
            critique,
            state.sub_questions,
        )

        if valid:
            target = derive_revision_target(critique)
            return critique, target

        if attempt == CRITIC_RETRY_LIMIT:
            break

        feedback = reason

    raise CritiqueValidationError(
        "Unable to produce a deterministically valid Critique after "
        f"{CRITIC_RETRY_LIMIT} validation retries."
    )


def critic_node(state: ResearchState) -> dict:
    """LangGraph-compatible Critic Agent node."""
    from src.core.llm import get_llm_service

    critique, target = critique_report(state, get_llm_service())

    return {
        "critique": critique,
        "revision_target": target,
    }
