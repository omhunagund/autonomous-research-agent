from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.agents.analysis_agent import (
    ANALYSIS_RETRY_LIMIT,
    AnalysisValidationError,
    _validate_internal_analysis,
    analyze_research,
    build_analysis,
    derive_confidence,
)
from src.models.schemas import (
    Analysis,
    AgreementLevel,
    ConfidenceLevel,
    Conflict,
    FinalReport,
    Gap,
    GapType,
    InternalAnalysis,
    InternalFindingAssessment,
    ResearchLimitation,
    ResearchState,
    Source,
    SubQuestionPlan,
)
from src.tools.page_fetcher import PageFetchError


def _source(citation_id: int) -> Source:
    return Source(
        citation_id=citation_id,
        title=f"Source {citation_id}",
        url=f"https://example.com/{citation_id}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Relevant evidence",
        search_queries=["research question"],
        content="Detailed evidence text " * 20,
    )


def _state() -> ResearchState:
    return ResearchState(
        user_topic="Artificial intelligence in healthcare",
        memory_context=None,
        sub_questions=[
            "What are the main clinical uses of artificial intelligence?",
            "What evidence supports improved patient outcomes?",
            "What are the major risks and limitations for deployment?",
            "Which regulatory developments currently affect deployment?",
        ],
        sources=[_source(1), _source(2), _source(3)],
        findings=[],
        conflicts=[],
        gaps=[],
        research_limitations=[],
        draft=None,
        critique=None,
        retry_count=0,
        revision_target=None,
        final_report=None,
    )


def _assessment(
    *,
    claim: str = "AI systems can improve selected clinical outcomes.",
    supporting_sources: list[int] | None = None,
    independent_sources: list[int] | None = None,
    agreement: AgreementLevel = AgreementLevel.AGREE,
    conflict: bool = False,
) -> InternalFindingAssessment:
    return InternalFindingAssessment(
        claim=claim,
        supporting_sources=supporting_sources or [1, 2],
        independent_sources=independent_sources or [1, 2],
        agreement=agreement,
        conflict=conflict,
    )


def test_confidence_is_high_for_two_independent_agreeing_sources() -> None:
    assessment = _assessment()
    assert derive_confidence(assessment) == ConfidenceLevel.HIGH


def test_confidence_is_medium_for_one_independent_agreeing_source() -> None:
    assessment = _assessment(
        supporting_sources=[1],
        independent_sources=[1],
    )
    assert derive_confidence(assessment) == ConfidenceLevel.MEDIUM


def test_confidence_is_low_for_partial_agreement() -> None:
    assessment = _assessment(
        agreement=AgreementLevel.PARTIAL,
    )
    assert derive_confidence(assessment) == ConfidenceLevel.LOW


def test_confidence_is_low_for_unclear_agreement() -> None:
    assessment = _assessment(
        agreement=AgreementLevel.UNCLEAR,
    )
    assert derive_confidence(assessment) == ConfidenceLevel.LOW


def test_conflict_always_overrides_source_count() -> None:
    assessment = _assessment(conflict=True)
    assert derive_confidence(assessment) == ConfidenceLevel.LOW


def test_validation_rejects_unknown_supporting_source() -> None:
    internal = InternalAnalysis(
        finding_assessments=[
            _assessment(supporting_sources=[1, 99], independent_sources=[1])
        ],
        conflicts=[],
        gaps=[],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "unknown supporting source" in reason


def test_validation_rejects_independent_source_outside_supporting_set() -> None:
    internal = InternalAnalysis(
        finding_assessments=[
            _assessment(supporting_sources=[1], independent_sources=[1, 2])
        ],
        conflicts=[],
        gaps=[],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "subset" in reason


def test_validation_rejects_conflict_without_matching_conflict_object() -> None:
    internal = InternalAnalysis(
        finding_assessments=[
            _assessment(
                supporting_sources=[1, 2],
                independent_sources=[1, 2],
                conflict=True,
            )
        ],
        conflicts=[],
        gaps=[],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "conflict=true" in reason


def test_validation_accepts_conflict_with_matching_related_sources() -> None:
    internal = InternalAnalysis(
        finding_assessments=[
            _assessment(
                supporting_sources=[1, 2],
                independent_sources=[1, 2],
                conflict=True,
            )
        ],
        conflicts=[
            Conflict(
                description="The sources report incompatible outcomes for the same scope.",
                related_sources=[2, 3],
            )
        ],
        gaps=[],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert valid
    assert reason == ""


def test_validation_rejects_conflict_with_fewer_than_two_sources() -> None:
    internal = InternalAnalysis(
        finding_assessments=[],
        conflicts=[
            Conflict(
                description="Insufficiently referenced conflict.",
                related_sources=[1],
            )
        ],
        gaps=[],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "at least two" in reason


def test_validation_rejects_unknown_gap_sub_question() -> None:
    internal = InternalAnalysis(
        finding_assessments=[],
        conflicts=[],
        gaps=[
            Gap(
                type=GapType.RESEARCH_GAP,
                description="The needed evidence was unavailable.",
                related_sub_question="This is not a current sub-question",
            )
        ],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "unknown sub-question" in reason


def test_validation_rejects_unknown_gap_claim() -> None:
    internal = InternalAnalysis(
        finding_assessments=[_assessment()],
        conflicts=[],
        gaps=[
            Gap(
                type=GapType.EVIDENCE_GAP,
                description="A related claim remains weak.",
                related_claim="Claim that was never generated",
            )
        ],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        _state().sources,
        _state().sub_questions,
    )

    assert not valid
    assert "unknown finding claim" in reason


def test_build_analysis_derives_confidence_and_keeps_conflicts_and_gaps() -> None:
    internal = InternalAnalysis(
        finding_assessments=[
            _assessment(
                claim="Claim one has two independent supporters.",
                supporting_sources=[1, 2],
                independent_sources=[1, 2],
            ),
            _assessment(
                claim="Claim two has one independent supporter.",
                supporting_sources=[3],
                independent_sources=[3],
            ),
        ],
        conflicts=[
            Conflict(
                description="A substantive disagreement exists.",
                related_sources=[1, 3],
            )
        ],
        gaps=[
            Gap(
                type=GapType.EVIDENCE_GAP,
                description="Evidence remains incomplete.",
                related_sub_question=None,
                related_claim="Claim two has one independent supporter.",
            )
        ],
    )

    result = build_analysis(internal)

    assert result.findings[0].confidence == ConfidenceLevel.HIGH
    assert result.findings[1].confidence == ConfidenceLevel.MEDIUM
    assert len(result.conflicts) == 1
    assert len(result.gaps) == 1


def test_analyze_research_uses_feedback_retry() -> None:
    state = _state()
    mock_llm = Mock()

    mock_llm.invoke_structured.side_effect = [
        InternalAnalysis(
            finding_assessments=[
                _assessment(supporting_sources=[1, 99], independent_sources=[1])
            ],
            conflicts=[],
            gaps=[],
        ),
        InternalAnalysis(
            finding_assessments=[
                _assessment(
                    claim="AI systems can improve selected clinical outcomes.",
                    supporting_sources=[1, 2],
                    independent_sources=[1, 2],
                )
            ],
            conflicts=[],
            gaps=[],
        ),
    ]

    result = analyze_research(state, mock_llm)

    assert result.findings[0].confidence == ConfidenceLevel.HIGH
    assert mock_llm.invoke_structured.call_count == 2


def test_analyze_research_fails_after_validation_retry_budget() -> None:
    state = _state()
    mock_llm = Mock()

    mock_llm.invoke_structured.return_value = InternalAnalysis(
        finding_assessments=[
            _assessment(supporting_sources=[1, 99], independent_sources=[1])
        ],
        conflicts=[],
        gaps=[],
    )

    with pytest.raises(AnalysisValidationError) as exc_info:
        analyze_research(state, mock_llm)

    assert "unknown supporting source IDs" in str(exc_info.value)
    assert exc_info.value.reason is not None

    assert mock_llm.invoke_structured.call_count == ANALYSIS_RETRY_LIMIT + 1


def test_validation_accepts_no_findings_when_research_has_only_gaps() -> None:
    state = _state()
    internal = InternalAnalysis(
        finding_assessments=[],
        conflicts=[],
        gaps=[
            Gap(
                type=GapType.RESEARCH_GAP,
                description="The sub-question could not be adequately researched.",
                related_sub_question=state.sub_questions[0],
                related_claim=None,
            )
        ],
    )

    valid, reason = _validate_internal_analysis(
        internal,
        state.sources,
        state.sub_questions,
    )

    assert valid
    assert reason == ""
