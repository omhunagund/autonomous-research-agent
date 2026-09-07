from datetime import datetime, timezone

import pytest

from src.agents.critic_agent import (
    CRITIC_RETRY_LIMIT,
    CriticInputValidationError,
    CritiqueValidationError,
    _validate_critique,
    _validate_reviewable_draft,
    critique_report,
    derive_revision_target,
)
from src.models.schemas import (
    Critique,
    CritiqueCheck,
    ConfidenceLevel,
    Conflict,
    FinalReport,
    Finding,
    Gap,
    GapType,
    ResearchState,
    Source,
)
from src.core.evidence_compaction import PROMPT_EVIDENCE_TOKEN_BUDGET
from src.agents.critic_agent import _source_context

def _source(citation_id: int) -> Source:
    return Source(
        citation_id=citation_id,
        title=f"Source {citation_id}",
        url=f"https://example.com/{citation_id}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Evidence",
        search_queries=["test"],
        content="Evidence content " * 30,
    )


def _state() -> ResearchState:
    return ResearchState(
        user_topic="AI in healthcare",
        memory_context=None,
        sub_questions=[
            "What evidence supports improved patient outcomes?",
            "What are the major deployment risks and limitations?",
        ],
        sources=[_source(1), _source(2), _source(3)],
        findings=[
            Finding(
                claim="AI systems can improve selected clinical outcomes.",
                supporting_sources=[1, 2],
                confidence=ConfidenceLevel.HIGH,
            ),
        ],
        conflicts=[
            Conflict(
                description="Sources report incompatible estimates for the same scope.",
                related_sources=[1, 3],
            )
        ],
        gaps=[
            Gap(
                type=GapType.EVIDENCE_GAP,
                description="Evidence remains incomplete for deployment risks.",
                related_sub_question="What are the major deployment risks and limitations?",
                related_claim=None,
            )
        ],
        research_limitations=[],
        draft=FinalReport(
            topic="AI in healthcare",
            executive_summary="AI systems can improve selected outcomes [1][2].",
            key_findings=[
                Finding(
                    claim="AI systems can improve selected clinical outcomes.",
                    supporting_sources=[1, 2],
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
            supporting_evidence=[
                "Studies report improvements in selected outcomes [1][2]."
            ],
            gaps=[
                Gap(
                    type=GapType.EVIDENCE_GAP,
                    description="Evidence remains incomplete for deployment risks.",
                    related_sub_question="What are the major deployment risks and limitations?",
                    related_claim=None,
                )
            ],
            gap_explanations=["Deployment-risk evidence remains incomplete."],
            conflicts=[
                Conflict(
                    description="Sources report incompatible estimates for the same scope.",
                    related_sources=[1, 3],
                )
            ],
            conflict_explanations=["The estimates differ materially [1][3]."],
            quality=None,
            references=[_source(1), _source(2), _source(3)],
        ),
        critique=None,
        retry_count=0,
        revision_target=None,
        final_report=None,
    )


def _critique(
    *,
    faithfulness=CritiqueCheck.PASS,
    coverage=CritiqueCheck.PASS,
    recency=CritiqueCheck.PASS,
    balance=CritiqueCheck.PASS,
    verdict=CritiqueCheck.PASS,
    issues=None,
) -> Critique:
    return Critique(
        faithfulness=faithfulness,
        coverage=coverage,
        recency=recency,
        balance=balance,
        verdict=verdict,
        issues=issues or [],
    )


def test_all_pass_derives_pass() -> None:
    critique = _critique()
    valid, reason = _validate_critique(critique)
    assert valid
    assert reason == ""
    assert derive_revision_target(critique) == "none"


def test_verdict_is_deterministically_consistent() -> None:
    critique = _critique(
        faithfulness=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.PASS,
        issues=["[WRITING] The summary overstates Finding 1."],
    )
    valid, reason = _validate_critique(critique)
    assert not valid
    assert "verdict is inconsistent" in reason


def test_fail_requires_check_compatible_issue() -> None:
    critique = _critique(
        faithfulness=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[RESEARCH] Missing evidence exists for another sub-question."],
    )
    valid, reason = _validate_critique(critique)
    assert not valid
    assert "faithfulness is FAIL" in reason


def test_info_cannot_explain_failure_by_itself() -> None:
    critique = _critique(
        coverage=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[INFO] The wording could be more concise."],
    )
    valid, reason = _validate_critique(critique)
    assert not valid
    assert "compatible" in reason or "INFO" in reason


def test_coverage_accepts_research_writing_or_analysis_categories() -> None:
    for category in ("RESEARCH", "WRITING", "ANALYSIS"):
        critique = _critique(
            coverage=CritiqueCheck.FAIL,
            verdict=CritiqueCheck.FAIL,
            issues=[f"[{category}] Sub-question 2 lacks meaningful treatment."],
        )
        valid, reason = _validate_critique(critique)
        assert valid, (category, reason)


def test_recency_rejects_analysis_category() -> None:
    critique = _critique(
        recency=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[ANALYSIS] Current evidence was not retrieved."],
    )
    valid, reason = _validate_critique(critique)
    assert not valid
    assert "recency is FAIL" in reason


def test_balance_accepts_analysis_category() -> None:
    critique = _critique(
        balance=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[ANALYSIS] A material perspective is absent from the baseline."],
    )
    valid, reason = _validate_critique(critique)
    assert valid
    assert reason == ""


def test_issue_prefix_requires_explanatory_text() -> None:
    critique = _critique(
        faithfulness=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[WRITING]"],
    )
    valid, reason = _validate_critique(critique)
    assert not valid
    assert "issue 1" in reason


def test_revision_target_prioritizes_research() -> None:
    critique = _critique(
        faithfulness=CritiqueCheck.FAIL,
        coverage=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=[
            "[WRITING] Finding 1 is overclaimed.",
            "[RESEARCH] Sub-question 2 lacks retrieved evidence.",
        ],
    )
    assert derive_revision_target(critique) == "research"


def test_revision_target_uses_writing_without_research() -> None:
    critique = _critique(
        faithfulness=CritiqueCheck.FAIL,
        balance=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=[
            "[WRITING] The summary overstates the evidence.",
            "[WRITING] The conflict is minimized.",
        ],
    )
    assert derive_revision_target(critique) == "writing"


def test_revision_target_surfaces_analysis() -> None:
    critique = _critique(
        balance=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.FAIL,
        issues=["[ANALYSIS] A material conflict is missing from the baseline."],
    )
    assert derive_revision_target(critique) == "analysis"


def test_structurally_invalid_draft_is_rejected() -> None:
    state = _state()
    state.draft = None

    with pytest.raises(CriticInputValidationError):
        _validate_reviewable_draft(state)


def test_structurally_invalid_draft_does_not_become_normal_critique() -> None:
    state = _state()
    state.draft = None

    with pytest.raises(CriticInputValidationError):
        critique_report(state, object())


def test_analysis_finding_drift_is_rejected_before_semantic_review() -> None:
    state = _state()
    state.draft.key_findings[0] = Finding(
        claim="Changed claim",
        supporting_sources=[1],
        confidence=ConfidenceLevel.HIGH,
    )

    with pytest.raises(CriticInputValidationError):
        _validate_reviewable_draft(state)


def test_missing_gap_explanation_is_rejected() -> None:
    state = _state()
    state.draft.gap_explanations = []

    with pytest.raises(CriticInputValidationError):
        _validate_reviewable_draft(state)


def test_duplicate_reference_is_rejected() -> None:
    state = _state()
    state.draft.references = [_source(1), _source(1)]

    with pytest.raises(CriticInputValidationError):
        _validate_reviewable_draft(state)

def test_critic_source_context_uses_compacted_evidence() -> None:
    state = _state()

    state.sources = [
        Source(
            citation_id=1,
            title="Large source",
            url="https://example.com/large",
            retrieved_at=datetime.now(timezone.utc),
            snippet="Relevant evidence",
            search_queries=["AI healthcare outcomes"],
            content=(
                "AI healthcare outcomes are improving.\n\n"
                + ("Unrelated filler paragraph. " * 20_000)
            ),
        ),
        Source(
            citation_id=2,
            title="Second source",
            url="https://example.com/second",
            retrieved_at=datetime.now(timezone.utc),
            snippet="Additional evidence",
            search_queries=["AI healthcare outcomes"],
            content="Additional evidence about healthcare outcomes.",
        ),
    ]

    original_contents = [source.content for source in state.sources]

    context = _source_context(state)

    assert "Source [1]" in context
    assert "Source [2]" in context

    # The original canonical evidence must remain unchanged.
    assert [source.content for source in state.sources] == original_contents

    # The huge filler should not make it into the prompt context.
    assert context.count("Unrelated filler paragraph.") < 100


def test_critic_uses_feedback_retry() -> None:
    state = _state()
    from unittest.mock import Mock

    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = [
        _critique(
            faithfulness=CritiqueCheck.FAIL,
            verdict=CritiqueCheck.PASS,
            issues=["[WRITING] Incorrectly consistent provisional response."],
        ),
        _critique(
            coverage=CritiqueCheck.FAIL,
            verdict=CritiqueCheck.FAIL,
            issues=["[RESEARCH] Sub-question 2 lacks sufficient retrieved evidence."],
        ),
    ]

    critique, target = critique_report(state, mock_llm)

    assert critique.coverage == CritiqueCheck.FAIL
    assert target == "research"
    assert mock_llm.invoke_structured.call_count == 2


def test_critic_fails_after_validation_retry_budget() -> None:
    state = _state()
    from unittest.mock import Mock

    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = _critique(
        faithfulness=CritiqueCheck.FAIL,
        verdict=CritiqueCheck.PASS,
        issues=["[WRITING] Inconsistent response."],
    )

    with pytest.raises(CritiqueValidationError):
        critique_report(state, mock_llm)

    assert mock_llm.invoke_structured.call_count == CRITIC_RETRY_LIMIT + 1


def test_info_is_allowed_with_all_pass() -> None:
    critique = _critique(
        issues=["[INFO] The report could use a shorter introductory sentence."]
    )
    valid, reason = _validate_critique(critique)
    assert valid
    assert reason == ""
