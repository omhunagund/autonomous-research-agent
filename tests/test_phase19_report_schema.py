from __future__ import annotations

from datetime import datetime, timezone

from src.agents.writing_agent import build_final_report
from src.models.schemas import (
    ConfidenceLevel,
    Finding,
    FinalReport,
    InternalReportDraft,
    Source,
    SupportingEvidence,
    ResearchState,
    QualityLevel,
    Gap,
    Conflict,
)


def _source(citation_id: int) -> Source:
    return Source(
        citation_id=citation_id,
        title=f"Source {citation_id}",
        url=f"https://example.com/{citation_id}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Snippet",
        search_queries=["query"],
        content="Content",
    )


def _state() -> ResearchState:
    finding = Finding(
        claim="Finding claim",
        supporting_sources=[1, 2],
        confidence=ConfidenceLevel.HIGH,
    )
    return ResearchState(
        report_id="report-1",
        user_topic="Test topic",
        memory_context=None,
        sub_questions=["Question"],
        sources=[_source(1), _source(2)],
        findings=[finding],
        conflicts=[],
        gaps=[],
        research_limitations=[],
        draft=None,
        critique=None,
        retry_count=0,
        revision_target=None,
        final_report=None,
    )


def _draft() -> InternalReportDraft:
    return InternalReportDraft(
        executive_summary="Summary [1]",
        finding_drafts=[
            {"text": "Finding [1][2]", "citation_ids": [1, 2]},
        ],
        evidence_drafts=[
            {
                "text": "Shared evidence [1][2]",
                "citation_ids": [1, 2],
                "related_finding_indices": [1],
            }
        ],
        gap_drafts=[],
        conflict_drafts=[],
        cited_source_ids=[1, 2],
    )


def test_build_final_report_preserves_structured_evidence() -> None:
    report = build_final_report(_state(), _draft())

    assert isinstance(report.supporting_evidence[0], SupportingEvidence)
    assert report.supporting_evidence[0].text == "Shared evidence [1][2]"
    assert report.supporting_evidence[0].citation_ids == [1, 2]
    assert report.supporting_evidence[0].related_finding_indices == [1]


def test_new_final_report_accepts_structured_evidence() -> None:
    report = FinalReport(
        topic="Topic",
        executive_summary="Summary",
        key_findings=[],
        supporting_evidence=[
            SupportingEvidence(
                text="Evidence",
                citation_ids=[1, 2],
                related_finding_indices=[1, 2],
            )
        ],
        gaps=[],
        conflicts=[],
        quality=QualityLevel.HIGH,
        references=[_source(1), _source(2)],
    )

    assert report.supporting_evidence[0].related_finding_indices == [1, 2]


def test_legacy_string_evidence_is_migrated_without_fabrication() -> None:
    report = FinalReport.model_validate(
        {
            "topic": "Topic",
            "executive_summary": "Summary",
            "key_findings": [],
            "supporting_evidence": ["Old evidence"],
            "gaps": [],
            "conflicts": [],
            "quality": "high",
            "references": [],
        }
    )

    entry = report.supporting_evidence[0]
    assert entry.text == "Old evidence"
    assert entry.citation_ids == []
    assert entry.related_finding_indices == []


def test_structured_evidence_is_not_modified_by_migration() -> None:
    payload = {
        "topic": "Topic",
        "executive_summary": "Summary",
        "key_findings": [],
        "supporting_evidence": [
            {
                "text": "New evidence",
                "citation_ids": [3],
                "related_finding_indices": [2],
            }
        ],
        "gaps": [],
        "conflicts": [],
        "quality": "medium",
        "references": [],
    }

    report = FinalReport.model_validate(payload)

    assert report.supporting_evidence[0].citation_ids == [3]
    assert report.supporting_evidence[0].related_finding_indices == [2]
