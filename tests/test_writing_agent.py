from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.agents.writing_agent import (
    WRITING_RETRY_LIMIT,
    WritingValidationError,
    _backfill_writer_citations,
    _has_invalid_bracket_citation,
    _parse_citations,
    _validate_internal_report_draft,
    _unique_in_order,
    _writing_messages,
    build_final_report,
    write_report,
)
from src.models.schemas import (
    ConfidenceLevel,
    Conflict,
    FinalReport,
    Finding,
    Gap,
    GapType,
    InternalReportDraft,
    ResearchState,
    Source,
    WrittenConflictDraft,
    WrittenEvidenceDraft,
    WrittenFindingDraft,
    WrittenGapDraft,
)


def _source(citation_id: int) -> Source:
    return Source(
        citation_id=citation_id,
        title=f"Source {citation_id}",
        url=f"https://example.com/{citation_id}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Relevant evidence",
        search_queries=["test question"],
        content="Detailed evidence " * 30,
    )


def _state() -> ResearchState:
    return ResearchState(
        report_id=None,
        user_topic="AI in healthcare",
        memory_context=None,
        sub_questions=["What evidence supports improved patient outcomes?"],
        sources=[_source(1), _source(2), _source(3)],
        findings=[
            Finding(
                claim="AI systems can improve selected clinical outcomes.",
                supporting_sources=[1, 2],
                confidence=ConfidenceLevel.HIGH,
            ),
            Finding(
                claim="Evidence remains limited in some settings.",
                supporting_sources=[3],
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ],
        conflicts=[
            Conflict(
                description="Two sources report incompatible estimates for the same scope.",
                related_sources=[1, 3],
            )
        ],
        gaps=[
            Gap(
                type=GapType.EVIDENCE_GAP,
                description="Evidence remains incomplete for one setting.",
                related_sub_question=None,
                related_claim="Evidence remains limited in some settings.",
            )
        ],
        research_limitations=[],
        draft=None,
        critique=None,
        retry_count=0,
        revision_target=None,
        final_report=None,
    )


def _draft(
    *,
    finding_drafts=None,
    evidence_drafts=None,
    gap_drafts=None,
    conflict_drafts=None,
    executive_summary="The evidence supports selected improvements [1].",
) -> InternalReportDraft:
    return InternalReportDraft(
        executive_summary=executive_summary,
        finding_drafts=(
            finding_drafts
            if finding_drafts is not None
            else [
                WrittenFindingDraft(
                    text="AI systems can improve selected outcomes [1][2].",
                ),
                WrittenFindingDraft(
                    text="Evidence remains limited in some settings [3].",
                ),
            ]
        ),
        evidence_drafts=(
            evidence_drafts
            if evidence_drafts is not None
            else [
                WrittenEvidenceDraft(
                    text="The available studies report improvements in selected outcomes [1][2].",
                    related_finding_indices=[1],
                ),
            ]
        ),
        gap_drafts=(
            gap_drafts
            if gap_drafts is not None
            else [
                WrittenGapDraft(
                    text="Coverage remains incomplete for one setting."
                )
            ]
        ),
        conflict_drafts=(
            conflict_drafts
            if conflict_drafts is not None
            else [
                WrittenConflictDraft(
                    text="The estimates differ materially across the cited sources [1][3]."
                )
            ]
        ),
    )


def test_parse_citations_and_unique_order() -> None:
    assert _parse_citations("Claim [1][3] and again [1].") == [1, 3, 1]
    assert _unique_in_order([1, 3, 1, 5, 3]) == [1, 3, 5]


def test_invalid_comma_citation_syntax_is_rejected() -> None:
    assert _has_invalid_bracket_citation("Claim [1, 3].")
    assert not _has_invalid_bracket_citation("Claim [1][3].")


def test_writing_prompt_requires_inline_citations_without_declared_citation_ids() -> None:
    messages = _writing_messages(_state())
    system_prompt = messages[0].content

    assert "Use only approved inline citation syntax: [1] or [1][3]." in system_prompt
    assert "Cite at least one source for every Finding." in system_prompt

    assert "citation_ids MUST exactly equal" not in system_prompt
    assert "citation_ids=[3, 1]" not in system_prompt


def test_writing_prompt_requires_cited_evidence_without_declared_citation_ids() -> None:
    messages = _writing_messages(_state())
    system_prompt = messages[0].content

    assert (
        "Supporting evidence is mandatory whenever at least one Finding exists."
        in system_prompt
    )
    assert (
        "When Findings are referenced, evidence citations must come only from "
        "those Findings' supporting_sources."
        in system_prompt
    )

    assert "citation_ids MUST exactly equal" not in system_prompt


def test_writing_prompt_requires_supporting_evidence_when_findings_exist() -> None:
    messages = _writing_messages(_state())
    system_prompt = messages[0].content

    assert (
        "Supporting evidence is mandatory whenever at least one Finding exists"
        in system_prompt
    )
    assert (
        "When Findings exist, generate at least one Evidence draft"
        in system_prompt
    )


def test_writing_prompt_requires_all_top_level_structured_fields() -> None:
    messages = _writing_messages(_state())
    system_prompt = messages[0].content

    assert "Always return every required top-level field" in system_prompt
    assert "Never omit a required field" in system_prompt
    assert (
        "When there are no Analysis conflicts, return conflict_drafts as an empty list"
        in system_prompt
    )


def test_writer_backfills_uncited_finding_and_logs_warning(caplog) -> None:
    state = _state()

    draft = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="AI systems can improve selected outcomes."
            ),
            WrittenFindingDraft(
                text="Evidence remains limited in some settings [3]."
            ),
        ]
    )

    with caplog.at_level("WARNING"):
        updated = _backfill_writer_citations(draft, state)

    assert updated.finding_drafts[0].text.endswith("[1]")
    assert (
        "citation backfill applied to finding draft 1"
        in caplog.text
    )


def test_writer_backfills_uncited_related_evidence_and_logs_warning(
    caplog,
) -> None:
    state = _state()

    draft = _draft(
        evidence_drafts=[
            WrittenEvidenceDraft(
                text="The available studies report improvements.",
                related_finding_indices=[2],
            )
        ]
    )

    with caplog.at_level("WARNING"):
        updated = _backfill_writer_citations(draft, state)

    assert updated.evidence_drafts[0].text.endswith("[3]")
    assert (
        "citation backfill applied to evidence draft 1"
        in caplog.text
    )


def test_writer_backfill_does_not_invent_citation_without_source() -> None:
    state = _state()
    state.sources = []
    state.findings = []
    state.gaps = []
    state.conflicts = []

    draft = InternalReportDraft(
        executive_summary="No reliable conclusion is established.",
        finding_drafts=[],
        evidence_drafts=[
            WrittenEvidenceDraft(
                text="Contextual evidence without a source.",
                related_finding_indices=[],
            )
        ],
        gap_drafts=[],
        conflict_drafts=[],
    )

    updated = _backfill_writer_citations(draft, state)

    assert "[1]" not in updated.evidence_drafts[0].text

    valid, reason = _validate_internal_report_draft(updated, state)

    assert not valid
    assert "must contain at least one citation" in reason


def test_writer_backfill_leaves_cited_text_unchanged() -> None:
    state = _state()
    draft = _draft()

    updated = _backfill_writer_citations(draft, state)

    assert updated.finding_drafts == draft.finding_drafts
    assert updated.evidence_drafts == draft.evidence_drafts


def test_validation_accepts_valid_internal_draft() -> None:
    state = _state()

    valid, reason = _validate_internal_report_draft(_draft(), state)

    assert valid
    assert reason == ""


def test_validation_rejects_finding_citation_outside_supporting_sources() -> None:
    state = _state()

    draft = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="Wrong source [3].",
            ),
            WrittenFindingDraft(
                text="Supported source [3].",
            ),
        ]
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "outside its approved supporting_sources" in reason


def test_validation_rejects_uncited_finding() -> None:
    state = _state()

    draft = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="A factual finding appears without a citation.",
            ),
            WrittenFindingDraft(
                text="Supported source [3].",
            ),
        ]
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "must contain at least one citation" in reason


def test_validation_rejects_evidence_citation_for_unrelated_finding() -> None:
    state = _state()

    draft = _draft(
        evidence_drafts=[
            WrittenEvidenceDraft(
                text="Evidence is described here [3].",
                related_finding_indices=[1],
            )
        ]
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "unrelated to its declared findings" in reason


def test_validation_accepts_unassigned_contextual_evidence() -> None:
    state = _state()

    draft = _draft(
        evidence_drafts=[
            WrittenEvidenceDraft(
                text="Source methodology is described here [3].",
                related_finding_indices=[],
            )
        ]
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert valid
    assert reason == ""


def test_validation_rejects_unknown_text_citation() -> None:
    state = _state()

    draft = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="Unsupported source claim [99].",
            ),
            WrittenFindingDraft(
                text="Supported source [3].",
            ),
        ]
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "unknown citation IDs" in reason


def test_validation_rejects_missing_gap_draft() -> None:
    state = _state()

    draft = _draft(gap_drafts=[])

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "gap_drafts count" in reason


def test_validation_rejects_missing_conflict_draft() -> None:
    state = _state()

    draft = _draft(conflict_drafts=[])

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "conflict_drafts count" in reason


def test_validation_allows_empty_findings_when_analysis_is_empty() -> None:
    state = _state()
    state.findings = []
    state.gaps = []
    state.conflicts = []

    draft = InternalReportDraft(
        executive_summary=(
            "The available research does not establish a reliable conclusion."
        ),
        finding_drafts=[],
        evidence_drafts=[],
        gap_drafts=[],
        conflict_drafts=[],
    )

    valid, reason = _validate_internal_report_draft(draft, state)

    assert valid
    assert reason == ""


def test_validation_rejects_upstream_finding_without_support() -> None:
    state = _state()

    state.findings[0] = Finding(
        claim=state.findings[0].claim,
        supporting_sources=[],
        confidence=ConfidenceLevel.HIGH,
    )

    valid, reason = _validate_internal_report_draft(_draft(), state)

    assert not valid
    assert "upstream Analysis integrity violation" in reason


def test_build_final_report_derives_finding_and_evidence_citations() -> None:
    state = _state()

    report = build_final_report(state, _draft())

    assert report.supporting_evidence[0].citation_ids == [1, 2]
    assert [source.citation_id for source in report.references] == [1, 2, 3]


def test_writer_draft_schema_does_not_expose_redundant_citation_fields() -> None:
    finding_schema = WrittenFindingDraft.model_json_schema()
    evidence_schema = WrittenEvidenceDraft.model_json_schema()
    report_schema = InternalReportDraft.model_json_schema()

    assert "citation_ids" not in finding_schema["properties"]
    assert "citation_ids" not in evidence_schema["properties"]
    assert "cited_source_ids" not in report_schema["properties"]


def test_build_final_report_preserves_analysis_and_filters_references() -> None:
    state = _state()

    report = build_final_report(state, _draft())

    assert report.topic == state.user_topic
    assert report.key_findings == state.findings
    assert report.gaps == state.gaps
    assert report.conflicts == state.conflicts
    assert [source.citation_id for source in report.references] == [1, 2, 3]
    assert report.gap_explanations
    assert report.conflict_explanations
    assert report.quality is None


def test_write_report_uses_feedback_retry() -> None:
    state = _state()
    mock_llm = Mock()

    invalid = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="Wrong source [3].",
            ),
            WrittenFindingDraft(
                text="Supported source [3].",
            ),
        ]
    )

    valid = _draft()

    mock_llm.invoke_structured.side_effect = [invalid, valid]

    result = write_report(state, mock_llm)

    assert isinstance(result, FinalReport)
    assert result.quality is None
    assert mock_llm.invoke_structured.call_count == 2


def test_write_report_fails_after_retry_budget() -> None:
    state = _state()
    mock_llm = Mock()

    mock_llm.invoke_structured.return_value = _draft(
        finding_drafts=[
            WrittenFindingDraft(
                text="Wrong source [3].",
            ),
            WrittenFindingDraft(
                text="Supported source [3].",
            ),
        ]
    )

    with pytest.raises(
        WritingValidationError,
        match="Final validation failure:.*outside its approved supporting_sources",
    ):
        write_report(state, mock_llm)

    assert mock_llm.invoke_structured.call_count == WRITING_RETRY_LIMIT + 1


def test_validation_rejects_empty_supporting_evidence_when_findings_exist() -> None:
    state = _state()
    draft = _draft(evidence_drafts=[])

    valid, reason = _validate_internal_report_draft(draft, state)

    assert not valid
    assert "supporting evidence" in reason