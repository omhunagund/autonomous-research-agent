"""Writing Agent for the autonomous research and report workflow.

The Writer is a controlled presentation layer:
- the LLM writes only prose and citation placement;
- Analysis-derived structured fields remain authoritative;
- deterministic validation enforces citation/reference integrity and
  positional preservation;
- report quality is deliberately left unset for the Critic/Orchestrator.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.evidence_compaction import compact_sources_for_prompt
from src.core.llm import LLMService
from src.models.schemas import (
    FinalReport,
    InternalReportDraft,
    ResearchState,
    Source,
    SupportingEvidence,
)

logger = logging.getLogger(__name__)

WRITING_RETRY_LIMIT = 2
VALID_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ANY_BRACKET_PATTERN = re.compile(r"\[([^\]]*)\]")


class WritingValidationError(RuntimeError):
    """Raised when a Writer response fails deterministic validation."""


def _parse_citations(text: str) -> list[int]:
    """Parse only approved [n] citation tokens from text."""
    return [int(match.group(1)) for match in VALID_CITATION_PATTERN.finditer(text)]


def _has_invalid_bracket_citation(text: str) -> bool:
    """Reject bracket groups that look like citations but are not exactly [n]."""
    for match in ANY_BRACKET_PATTERN.finditer(text):
        content = match.group(1).strip()
        if not content:
            continue
        if content.isdigit():
            continue
        # The Writer's citation syntax is deliberately restricted to [n].
        if "," in content or re.fullmatch(r"\d+(?:\s+\d+)+", content):
            return True
    return False


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []

    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)

    return result


def _validate_text_citations(
    *,
    text: str,
    valid_source_ids: set[int],
    context: str,
) -> tuple[bool, str]:
    if _has_invalid_bracket_citation(text):
        return False, (
            f"{context} contains invalid citation syntax; use only [n] or [n][m]"
        )

    parsed = _unique_in_order(_parse_citations(text))

    unknown = sorted(set(parsed) - valid_source_ids)
    if unknown:
        return False, f"{context} contains unknown citation IDs: {unknown}"

    return True, ""


def _all_text_fields(draft: InternalReportDraft) -> list[str]:
    texts = [draft.executive_summary]
    texts.extend(item.text for item in draft.finding_drafts)
    texts.extend(item.text for item in draft.evidence_drafts)
    texts.extend(item.text for item in draft.gap_drafts)
    texts.extend(item.text for item in draft.conflict_drafts)
    return texts

def _backfill_writer_citations(
    draft: InternalReportDraft,
    state: ResearchState,
) -> InternalReportDraft:
    """
    Backfill missing Writer citations from authoritative Analysis sources.

    This is a safety net, not a replacement for normal Writer citation
    behavior. Every backfill is logged so the intervention remains visible.
    Canonical Analysis fields are never modified.
    """
    finding_drafts = list(draft.finding_drafts)
    evidence_drafts = list(draft.evidence_drafts)

    changed = False

    for index, (finding, finding_draft) in enumerate(
        zip(state.findings, finding_drafts),
        start=1,
    ):
        if _parse_citations(finding_draft.text):
            continue

        if not finding.supporting_sources:
            continue

        citation_id = finding.supporting_sources[0]
        finding_drafts[index - 1] = finding_draft.model_copy(
            update={
                "text": f"{finding_draft.text.rstrip()} [{citation_id}]",
            }
        )
        changed = True

        logger.warning(
            "Writer citation backfill applied to finding draft %d; "
            "appended approved source citation [%d].",
            index,
            citation_id,
        )

    for index, evidence_draft in enumerate(evidence_drafts, start=1):
        if _parse_citations(evidence_draft.text):
            continue

        citation_id: int | None = None

        if evidence_draft.related_finding_indices:
            first_finding_index = evidence_draft.related_finding_indices[0]
            finding = state.findings[first_finding_index - 1]

            if finding.supporting_sources:
                citation_id = finding.supporting_sources[0]
        elif state.sources:
            citation_id = state.sources[0].citation_id

        if citation_id is None:
            logger.warning(
                "Writer citation backfill could not be applied to evidence "
                "draft %d because no approved source was available.",
                index,
            )
            continue

        evidence_drafts[index - 1] = evidence_draft.model_copy(
            update={
                "text": f"{evidence_draft.text.rstrip()} [{citation_id}]",
            }
        )
        changed = True

        logger.warning(
            "Writer citation backfill applied to evidence draft %d; "
            "appended approved source citation [%d].",
            index,
            citation_id,
        )

    if not changed:
        return draft

    return draft.model_copy(
        update={
            "finding_drafts": finding_drafts,
            "evidence_drafts": evidence_drafts,
        },
        deep=True,
    )


def _validate_internal_report_draft(
    draft: InternalReportDraft,
    state: ResearchState,
) -> tuple[bool, str]:
    source_ids = {source.citation_id for source in state.sources}
    finding_count = len(state.findings)
    gap_count = len(state.gaps)
    conflict_count = len(state.conflicts)

    # Upstream Analysis invariant: every Finding must already be evidence-grounded.
    for finding_index, finding in enumerate(state.findings, start=1):
        if not finding.supporting_sources:
            return False, (
                f"Analysis finding {finding_index} has no supporting_sources; "
                "Writer refuses to repair an upstream Analysis integrity violation"
            )
        unknown = sorted(set(finding.supporting_sources) - source_ids)
        if unknown:
            return False, (
                f"Analysis finding {finding_index} references unknown source IDs: "
                f"{unknown}"
            )

    if len(draft.finding_drafts) != finding_count:
        return False, "finding_drafts count must equal Analysis.findings count"

    if len(draft.gap_drafts) != gap_count:
        return False, "gap_drafts count must equal Analysis.gaps count"

    if len(draft.conflict_drafts) != conflict_count:
        return False, "conflict_drafts count must equal Analysis.conflicts count"

    if finding_count and not draft.evidence_drafts:
        return False, (
            "supporting evidence must contain at least one entry when findings exist"
        )

    for index, (finding, finding_draft) in enumerate(
        zip(state.findings, draft.finding_drafts),
        start=1,
    ):
        if not finding_draft.text.strip():
            return False, f"finding draft {index} has empty text"

        valid, reason = _validate_text_citations(
            text=finding_draft.text,
            valid_source_ids=source_ids,
            context=f"finding draft {index}",
        )
        if not valid:
            return False, reason

        finding_citation_ids = _unique_in_order(
            _parse_citations(finding_draft.text)
        )

        if not finding_citation_ids:
            return False, f"finding draft {index} must contain at least one citation"

        unauthorized = sorted(
            set(finding_citation_ids) - set(finding.supporting_sources)
        )
        if unauthorized:
            return False, (
                f"finding draft {index} cites sources outside its approved "
                f"supporting_sources: {unauthorized}"
            )

    for index, evidence_draft in enumerate(draft.evidence_drafts, start=1):
        if not evidence_draft.text.strip():
            return False, f"evidence draft {index} has empty text"

        if not _parse_citations(evidence_draft.text):
            return False, (
                f"evidence draft {index} must contain at least one citation"
        )

        valid, reason = _validate_text_citations(
            text=evidence_draft.text,
            valid_source_ids=source_ids,
            context=f"evidence draft {index}",
        )
        if not valid:
            return False, reason

        for finding_index in evidence_draft.related_finding_indices:
            if not 1 <= finding_index <= finding_count:
                return False, (
                    f"evidence draft {index} references invalid finding index "
                    f"{finding_index}"
                )

        if evidence_draft.related_finding_indices:
            allowed: set[int] = set()
            for finding_index in evidence_draft.related_finding_indices:
                allowed.update(state.findings[finding_index - 1].supporting_sources)

            evidence_citation_ids = _unique_in_order(
                _parse_citations(evidence_draft.text)
            )

            unauthorized = sorted(
                set(evidence_citation_ids) - allowed
            )
            if unauthorized:
                return False, (
                    f"evidence draft {index} cites sources unrelated to its "
                    f"declared findings: {unauthorized}"
                )

    for index, gap_draft in enumerate(draft.gap_drafts, start=1):
        if not gap_draft.text.strip():
            return False, f"gap draft {index} has empty text"

    for index, conflict_draft in enumerate(draft.conflict_drafts, start=1):
        if not conflict_draft.text.strip():
            return False, f"conflict draft {index} has empty text"

    report_citations = _unique_in_order(
        citation
        for text in _all_text_fields(draft)
        for citation in _parse_citations(text)
    )

    unknown_report_citations = sorted(set(report_citations) - source_ids)
    if unknown_report_citations:
        return False, (
            "Writer-generated text contains unknown citation IDs: "
            f"{unknown_report_citations}"
        )

    return True, ""


def _writing_messages(
    state: ResearchState,
    feedback: str | None = None,
    correction_issues: list[str] | None = None,
) -> list:
    system = (
        "You are the Writing Agent in an autonomous research and report "
        "system. You are a controlled presentation layer, not a new research "
        "or reasoning agent. Use only the supplied Analysis and freshly "
        "retrieved Source evidence. Never introduce a new factual claim "
        "absent from Analysis. Never change, reorder, remove, or reinterpret "
        "Analysis Findings, Gaps, or Conflicts. For each Finding, generate "
        "only the written text for that Finding. The original claim, "
        "supporting_sources, and confidence are immutable and are not writable. "
        "Use only approved inline citation syntax: [1] or [1][3]. "
        "Cite at least one source for every Finding. Cite each factual "
        "statement at the smallest practical unit. Supporting evidence is "
        "mandatory whenever at least one Finding exists. When Findings exist, "
        "generate at least one Evidence draft. Evidence drafts may reference "
        "multiple Findings with 1-based related_finding_indices or may have "
        "an empty relationship for contextual evidence. When Findings are "
        "referenced, evidence citations must come only from those Findings' "
        "supporting_sources. Represent every Analysis gap and conflict exactly "
        "once, in order. Do not create new gaps or conflicts. Keep their "
        "structured meaning unchanged. Material gaps and conflicts must remain "
        "visible. Always return every required top-level field in the "
        "structured response: executive_summary, finding_drafts, "
        "evidence_drafts, gap_drafts, and conflict_drafts. Never omit a "
        "required field. When there are no Analysis conflicts, return "
        "conflict_drafts as an empty list. When there are no Analysis gaps, "
        "return gap_drafts as an empty list. When there are no Analysis "
        "Findings, return finding_drafts and evidence_drafts as empty lists. "
        "Executive-summary factual statements require citations. Memory is "
        "planning context only and never evidence. Do not assign report quality."
    )

    findings_context = "\n".join(
        (
            f"{index}. claim={finding.claim!r}; "
            f"supporting_sources={finding.supporting_sources}; "
            f"confidence={finding.confidence.value}"
        )
        for index, finding in enumerate(state.findings, start=1)
    ) or "No Findings."

    gaps_context = "\n".join(
        (
            f"{index}. type={gap.type.value}; description={gap.description!r}; "
            f"related_sub_question={gap.related_sub_question!r}; "
            f"related_claim={gap.related_claim!r}"
        )
        for index, gap in enumerate(state.gaps, start=1)
    ) or "No Gaps."

    conflicts_context = "\n".join(
        (
            f"{index}. description={conflict.description!r}; "
            f"related_sources={conflict.related_sources}"
        )
        for index, conflict in enumerate(state.conflicts, start=1)
    ) or "No Conflicts."

    compacted_sources = compact_sources_for_prompt(
        state.sources,
        state.sub_questions,
    )

    sources_context = "\n\n---\n\n".join(
        (
            f"Source [{source.citation_id}]\n"
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Content:\n{source.content}"
        )
        for source in compacted_sources
    ) or "No sources."

    user = (
        f"Research topic:\n{state.user_topic}\n\n"
        f"Sub-questions:\n"
        + "\n".join(f"- {question}" for question in state.sub_questions)
        + "\n\n"
        f"Analysis Findings (preserve order and immutable fields):\n{findings_context}\n\n"
        f"Analysis Gaps (preserve order and fields):\n{gaps_context}\n\n"
        f"Analysis Conflicts (preserve order and fields):\n{conflicts_context}\n\n"
        f"Fresh source evidence:\n{sources_context}"
    )

    if feedback:
        user += (
            "\n\nThe previous structured Writer response failed deterministic "
            f"validation. Fix this exact issue: {feedback}. Return only the "
            "corrected structured draft."
        )

    if correction_issues:
        user += (
            "\n\nCritic-driven Writing correction instructions. Correct these latest "
            "issues only while preserving the immutable Analysis Findings, "
            "Gaps, Conflicts, confidence values, and source relationships:\n"
            + "\n".join(f"- {issue}" for issue in correction_issues)
        )

    return [SystemMessage(content=system), HumanMessage(content=user)]


def build_final_report(
    state: ResearchState,
    draft: InternalReportDraft,
) -> FinalReport:
    """Assemble a provisional report; quality intentionally remains unset."""
    source_by_id = {source.citation_id: source for source in state.sources}

    cited_source_ids = _unique_in_order(
        citation
        for text in _all_text_fields(draft)
        for citation in _parse_citations(text)
    )

    references = [
        source_by_id[citation_id]
        for citation_id in sorted(cited_source_ids)
    ]

    return FinalReport(
        topic=state.user_topic,
        executive_summary=draft.executive_summary.strip(),
        key_findings=list(state.findings),
        supporting_evidence=[
            SupportingEvidence(
                text=item.text.strip(),
                citation_ids=_unique_in_order(
                    _parse_citations(item.text)
                ),
                related_finding_indices=list(item.related_finding_indices),
            )
            for item in draft.evidence_drafts
        ],
        gaps=list(state.gaps),
        gap_explanations=[item.text.strip() for item in draft.gap_drafts],
        conflicts=list(state.conflicts),
        conflict_explanations=[
            item.text.strip() for item in draft.conflict_drafts
        ],
        quality=None,
        references=references,
    )


def write_report(
    state: ResearchState,
    llm_service: LLMService,
    correction_issues: list[str] | None = None,
) -> FinalReport:
    """Generate and validate a Writer draft, then assemble a provisional report."""
    feedback: str | None = None

    for attempt in range(WRITING_RETRY_LIMIT + 1):
        draft = llm_service.invoke_structured(
            _writing_messages(state, feedback, correction_issues),
            InternalReportDraft,
        )

        draft = _backfill_writer_citations(draft, state)

        valid, reason = _validate_internal_report_draft(draft, state)
        if valid:
            return build_final_report(state, draft)

        if attempt == WRITING_RETRY_LIMIT:
            break

        feedback = reason

    raise WritingValidationError(
        "Unable to produce a valid report draft after "
        f"{WRITING_RETRY_LIMIT} validation retries. "
        f"Final validation failure: {reason}"
    )


def writing_node(state: ResearchState) -> dict:
    """LangGraph-compatible Writing Agent node."""
    from src.core.llm import get_llm_service

    report = write_report(state, get_llm_service())

    return {"draft": report}
