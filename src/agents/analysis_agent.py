"""Analysis Agent for the autonomous research workflow.

The Analysis Agent converts retrieved source evidence into atomic, grounded
findings, contextual conflicts, and research/evidence gaps.

LLM responsibilities:
- synthesize atomic claims from source evidence
- identify which sources substantively support each claim
- identify semantically independent supporting sources
- judge agreement as agree/partial/unclear
- identify substantive conflicts using scope-aware reasoning
- classify research_gap vs evidence_gap

Deterministic responsibilities:
- validate citation/reference integrity
- validate independence subset relationships
- validate conflict-to-assessment linkage
- derive Finding confidence without accepting an LLM confidence label
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.evidence_compaction import compact_sources_for_prompt
from src.core.llm import LLMService
from src.models.schemas import (
    Analysis,
    ConfidenceLevel,
    Conflict,
    Gap,
    InternalAnalysis,
    InternalFindingAssessment,
    ResearchState,
    Source,
)


ANALYSIS_RETRY_LIMIT = 2


class AnalysisValidationError(RuntimeError):
    """Raised when the LLM analysis cannot satisfy deterministic validation."""


def _source_context(
    sources: list[Source],
    sub_questions: list[str],
) -> str:
    if not sources:
        return "No usable sources were retrieved."

    compacted_sources = compact_sources_for_prompt(
        sources,
        sub_questions,
    )

    if not compacted_sources:
        return "No source evidence fit within the analysis context budget."

    blocks: list[str] = []

    for source in compacted_sources:
        blocks.append(
            f"Source [{source.citation_id}]\n"
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Retrieved at: {source.retrieved_at.isoformat()}\n"
            f"Search queries: {', '.join(source.search_queries)}\n"
            f"Snippet: {source.snippet}\n"
            f"Content:\n{source.content}"
        )

    return "\n\n---\n\n".join(blocks)


def _limitation_context(state: ResearchState) -> str:
    if not state.research_limitations:
        return "No research limitations were recorded."

    return "\n".join(
        (
            f"- Sub-question: {limitation.sub_question}\n"
            f"  Usable unique URLs: {limitation.usable_source_count}\n"
            f"  Failed attempted candidates: {limitation.failed_candidate_count}\n"
            f"  Description: {limitation.description}"
        )
        for limitation in state.research_limitations
    )


def _memory_context(state: ResearchState) -> str:
    if state.memory_context is None or not state.memory_context.matches:
        return "No prior-report memory matches were found."

    lines: list[str] = []
    for match in state.memory_context.matches:
        lines.append(
            f"Prior report: {match.topic} (report_id={match.report_id})\n"
            f"Prior findings: {match.key_findings}\n"
            f"Prior gaps: {match.gaps}\n"
            f"Prior conflicts: {match.conflicts}"
        )

    return "\n\n".join(lines)


def _analysis_messages(
    state: ResearchState,
    feedback: str | None = None,
) -> list:
    system = (
        "You are the Analysis Agent in an autonomous research and report "
        "system. Analyze only the freshly retrieved web evidence supplied in "
        "the current run. Produce atomic, evidence-grounded claims; do not "
        "assign confidence labels. For each finding, identify only sources "
        "that substantively support the exact claim, then identify which of "
        "those sources are semantically independent. Independence is a "
        "semantic judgment: exclude obvious syndication, direct quotation, "
        "shared press releases, republication, or clear common-origin "
        "relationships, but do not treat shared topics or similar terminology "
        "as non-independent by itself. Set agreement to agree only when the "
        "independent evidence substantively supports the same claim. Use "
        "partial when support is incomplete or only partly aligned, and "
        "unclear when the evidence cannot establish the claim. Set conflict "
        "true only for substantive incompatibility after considering "
        "timeframe, geography, population/segment, methodology, measurement "
        "definition, and scope. Record such conflicts in the conflicts list. "
        "Do not create duplicate competing findings merely to represent both "
        "sides of one substantive conflict. Classify gaps as research_gap "
        "when retrieval did not obtain enough usable evidence for an important "
        "sub-question, and evidence_gap when evidence exists but remains "
        "weak, incomplete, ambiguous, or insufficient for a reliable claim. "
        "Research limitations are facts from retrieval, not automatic gaps. "
        "Do not use prior memory reports as evidence for current findings. "
        "Every current finding must be supported by current Source objects."
    )

    user = (
        f"Current research topic:\n{state.user_topic}\n\n"
        f"Sub-questions:\n"
        + "\n".join(f"- {question}" for question in state.sub_questions)
        + "\n\n"
        f"Current retrieved sources:\n"
        f"{_source_context(state.sources, state.sub_questions)}\n\n"
        f"Research limitations:\n{_limitation_context(state)}\n\n"
        f"Prior memory context (planning context only, never evidence):\n"
        f"{_memory_context(state)}"
    )

    if feedback:
        user += (
            "\n\nThe previous structured analysis failed deterministic "
            f"validation. Correct this exact issue: {feedback}. Return only "
            "the corrected structured analysis."
        )

    return [SystemMessage(content=system), HumanMessage(content=user)]


def _validate_internal_analysis(
    analysis: InternalAnalysis,
    sources: list[Source],
    sub_questions: list[str],
) -> tuple[bool, str]:
    source_ids = {source.citation_id for source in sources}
    finding_claims = {item.claim for item in analysis.finding_assessments}

    for position, assessment in enumerate(
        analysis.finding_assessments,
        start=1,
    ):
        if not assessment.claim.strip():
            return False, f"finding assessment {position} has an empty claim"

        if not assessment.supporting_sources:
            return False, (
                f"finding assessment {position} must contain at least one "
                "supporting source"
            )

        unknown_supporting = sorted(
            set(assessment.supporting_sources) - source_ids
        )
        if unknown_supporting:
            return False, (
                f"finding assessment {position} references unknown supporting "
                f"source IDs: {unknown_supporting}"
            )

        unknown_independent = sorted(
            set(assessment.independent_sources) - source_ids
        )
        if unknown_independent:
            return False, (
                f"finding assessment {position} references unknown independent "
                f"source IDs: {unknown_independent}"
            )

        if not set(assessment.independent_sources).issubset(
            set(assessment.supporting_sources)
        ):
            return False, (
                f"finding assessment {position} has independent sources that "
                "are not a subset of supporting sources"
            )

        if assessment.conflict and not any(
            set(conflict.related_sources)
            & set(assessment.supporting_sources)
            for conflict in analysis.conflicts
        ):
            return False, (
                f"finding assessment {position} marks conflict=true but no "
                "Conflict object references a related supporting source"
            )

    for position, conflict in enumerate(analysis.conflicts, start=1):
        if len(conflict.related_sources) < 2:
            return False, (
                f"conflict {position} must reference at least two source IDs"
            )

        unknown = sorted(set(conflict.related_sources) - source_ids)
        if unknown:
            return False, (
                f"conflict {position} references unknown source IDs: {unknown}"
            )

    for position, gap in enumerate(analysis.gaps, start=1):
        if gap.related_sub_question is not None:
            if gap.related_sub_question not in sub_questions:
                return False, (
                    f"gap {position} references an unknown sub-question"
                )

        if gap.related_claim is not None:
            if gap.related_claim not in finding_claims:
                return False, f"gap {position} references an unknown finding claim"

    return True, ""


def derive_confidence(
    assessment: InternalFindingAssessment,
) -> ConfidenceLevel:
    """Derive confidence solely from deterministic approved rules."""
    if assessment.conflict:
        return ConfidenceLevel.LOW

    if assessment.agreement.value != "agree":
        return ConfidenceLevel.LOW

    independent_count = len(assessment.independent_sources)

    if independent_count >= 2:
        return ConfidenceLevel.HIGH

    if independent_count == 1:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def build_analysis(
    internal: InternalAnalysis,
) -> Analysis:
    """Convert validated internal semantic assessments to canonical Analysis."""
    findings = [
        {
            "claim": assessment.claim,
            "supporting_sources": assessment.supporting_sources,
            "confidence": derive_confidence(assessment),
        }
        for assessment in internal.finding_assessments
    ]

    return Analysis(
        findings=findings,
        conflicts=internal.conflicts,
        gaps=internal.gaps,
    )


def analyze_research(
    state: ResearchState,
    llm_service: LLMService,
) -> Analysis:
    """Run the Analysis Agent with deterministic validation retries."""
    feedback: str | None = None

    for attempt in range(ANALYSIS_RETRY_LIMIT + 1):
        internal = llm_service.invoke_structured(
            _analysis_messages(state, feedback),
            InternalAnalysis,
        )

        valid, reason = _validate_internal_analysis(
            internal,
            state.sources,
            state.sub_questions,
        )

        if valid:
            return build_analysis(internal)

        if attempt == ANALYSIS_RETRY_LIMIT:
            break

        feedback = reason

    raise AnalysisValidationError(
        "Unable to produce a valid analysis after "
        f"{ANALYSIS_RETRY_LIMIT} validation retries."
    )


def analysis_node(state: ResearchState) -> dict:
    """LangGraph-compatible Analysis Agent node."""
    from src.core.llm import get_llm_service

    analysis = analyze_research(state, get_llm_service())

    return {
        "findings": analysis.findings,
        "conflicts": analysis.conflicts,
        "gaps": analysis.gaps,
    }
