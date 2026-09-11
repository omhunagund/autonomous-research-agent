
import pytest

from src.models.schemas import (
    Analysis,
    Critique,
    CritiqueCheck,
    FinalReport,
    QualityLevel,
    ResearchState,
)
from src.workflow import (
    _finalize_analysis,
    _finalize_pass,
    _finalize_unresolved_node,
    _research_correction_node,
    _research_node,
    _route_after_critic,
)
from src.agents.orchestrator import MAX_CORRECTION_CYCLES


def _report(topic: str = "Test topic") -> FinalReport:
    return FinalReport(
        topic=topic,
        executive_summary="A concise summary.",
        key_findings=[],
        supporting_evidence=[],
        gaps=[],
        gap_explanations=[],
        conflicts=[],
        conflict_explanations=[],
        quality=None,
        references=[],
        analysis_issues=[],
        unresolved_issues=[],
    )


def _state(
    *,
    critique: Critique | None,
    retry_count: int = 0,
    revision_target: str | None = None,
) -> ResearchState:
    return ResearchState(
        report_id="550e8400-e29b-41d4-a716-446655440000",
        user_topic="AI in healthcare",
        memory_context=None,
        sub_questions=[],
        sources=[],
        findings=[],
        conflicts=[],
        gaps=[],
        research_limitations=[],
        draft=_report(),
        critique=critique,
        retry_count=retry_count,
        revision_target=revision_target,
        final_report=None,
    )


def _critique(
    *,
    verdict: CritiqueCheck,
    issues: list[str],
) -> Critique:
    return Critique(
        faithfulness=CritiqueCheck.PASS,
        coverage=CritiqueCheck.PASS,
        recency=CritiqueCheck.PASS,
        balance=CritiqueCheck.PASS,
        verdict=verdict,
        issues=issues,
    )


def test_route_passes_to_finalization() -> None:
    state = _state(
        critique=_critique(verdict=CritiqueCheck.PASS, issues=[]),
    )

    assert _route_after_critic(state) == "pass"


def test_finalize_pass_is_high_quality_on_initial_attempt() -> None:
    state = _state(
        critique=_critique(verdict=CritiqueCheck.PASS, issues=[]),
        retry_count=0,
    )

    result = _finalize_pass(state)

    assert result["final_report"] is not None
    assert result["final_report"].quality is QualityLevel.HIGH
    assert result["revision_target"] == "none"


def test_finalize_pass_is_medium_quality_after_correction() -> None:
    state = _state(
        critique=_critique(verdict=CritiqueCheck.PASS, issues=[]),
        retry_count=1,
    )

    result = _finalize_pass(state)

    assert result["final_report"] is not None
    assert result["final_report"].quality is QualityLevel.MEDIUM
    assert result["revision_target"] == "none"


def test_analysis_only_failure_routes_to_analysis_finalize() -> None:
    issues = [
        "[ANALYSIS] The finding does not adequately reconcile the evidence.",
        "[INFO] The report could use clearer context.",
    ]
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=issues),
        revision_target="analysis",
    )

    assert _route_after_critic(state) == "analysis_finalize"


def test_analysis_finalize_sets_low_quality_and_preserves_analysis_issues() -> None:
    issues = [
        "[ANALYSIS] The finding does not adequately reconcile the evidence.",
        "[INFO] The report could use clearer context.",
    ]
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=issues),
        revision_target="analysis",
    )

    result = _finalize_analysis(state)

    assert result["final_report"] is not None
    assert result["final_report"].quality is QualityLevel.LOW
    assert result["final_report"].analysis_issues == [
        issues[0]
    ]
    assert result["final_report"].unresolved_issues == []
    assert result["revision_target"] == "analysis"


def test_research_failure_routes_to_research_correction() -> None:
    issue = '[RESEARCH] Sub-question: "What evidence exists?" lacks current support.'
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=[issue]),
        retry_count=0,
        revision_target="research",
    )

    assert _route_after_critic(state) == "research"


def test_writing_failure_routes_to_writing_correction() -> None:
    issue = "[WRITING] The report misstates the source evidence."
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=[issue]),
        retry_count=0,
        revision_target="writing",
    )

    assert _route_after_critic(state) == "writing"


def test_research_failure_at_retry_limit_routes_to_unresolved() -> None:
    issue = '[RESEARCH] Sub-question: "What evidence exists?" still lacks support.'
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=[issue]),
        retry_count=MAX_CORRECTION_CYCLES,
        revision_target="research",
    )

    assert _route_after_critic(state) == "unresolved_finalize"


def test_writing_failure_at_retry_limit_routes_to_unresolved() -> None:
    issue = "[WRITING] The report still misstates source evidence."
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=[issue]),
        retry_count=MAX_CORRECTION_CYCLES,
        revision_target="writing",
    )

    assert _route_after_critic(state) == "unresolved_finalize"


def test_mixed_research_and_analysis_failure_does_not_use_analysis_only_finalize() -> None:
    issues = [
        '[RESEARCH] Sub-question: "What evidence exists?" lacks support.',
        "[ANALYSIS] The conclusion does not sufficiently reconcile the evidence.",
    ]
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=issues),
        retry_count=0,
        revision_target="research",
    )

    assert _route_after_critic(state) == "research"


def test_non_actionable_failed_critique_routes_to_unresolved() -> None:
    state = _state(
        critique=_critique(
            verdict=CritiqueCheck.FAIL,
            issues=["[INFO] The report could be clearer."],
        ),
        retry_count=0,
        revision_target="none",
    )

    assert _route_after_critic(state) == "unresolved_finalize"


def test_missing_critique_raises_route_error() -> None:
    state = _state(critique=None)

    with pytest.raises(RuntimeError, match="Critic route requires a Critique"):
        _route_after_critic(state)


def test_unresolved_finalize_sets_low_quality_and_unresolved_issues() -> None:
    issues = [
        '[RESEARCH] Sub-question: "What evidence exists?" still lacks support.',
        "[WRITING] The report still misstates source evidence.",
    ]
    state = _state(
        critique=_critique(verdict=CritiqueCheck.FAIL, issues=issues),
        retry_count=MAX_CORRECTION_CYCLES,
        revision_target="research",
    )

    result = _finalize_unresolved_node(state)

    assert result["final_report"] is not None
    assert result["final_report"].quality is QualityLevel.LOW
    assert result["final_report"].analysis_issues == []
    assert result["final_report"].unresolved_issues == issues
    assert result["revision_target"] == "none"


def test_research_correction_logs_local_query_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub_question = "What evidence supports improved patient outcomes?"
    issue = (
        f'[RESEARCH] Sub-question: "{sub_question}" needs current evidence.'
    )

    state = ResearchState(
        report_id="550e8400-e29b-41d4-a716-446655440000",
        user_topic="AI in healthcare",
        memory_context=None,
        sub_questions=[sub_question],
        sources=[],
        findings=[],
        conflicts=[],
        gaps=[],
        research_limitations=[],
        draft=_report(),
        critique=_critique(
            verdict=CritiqueCheck.FAIL,
            issues=[issue],
        ),
        retry_count=0,
        revision_target="research",
        final_report=None,
    )

    updated_state = state.model_copy(update={"retry_count": 1})

    monkeypatch.setattr(
        "src.workflow.run_research_correction",
        lambda state, research_issues, llm_service: (
            updated_state,
            True,
            False,
        ),
    )

    monkeypatch.setattr(
        "src.workflow.analyze_research",
        lambda state, llm_service: Analysis(
            findings=[],
            conflicts=[],
            gaps=[],
        ),
    )

    events: list[dict] = []

    def recorder(*args, **kwargs) -> None:
        events.append(kwargs)

    result = _research_correction_node(state, recorder)

    assert result["retry_count"] == 1

    recovery_events = [
        event
        for event in events
        if event["event_type"] == "correction_query_generation_recovered"
    ]

    assert len(recovery_events) == 1
    assert recovery_events[0]["stage"] == "research"
    assert recovery_events[0]["attempt"] == 1
    assert (
        "local recovery from malformed structured output"
        in recovery_events[0]["message"]
    )


def test_research_logs_search_selection_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(
        critique=None,
        retry_count=0,
        revision_target=None,
    )

    updated_state = state.model_copy(
        update={
            "sub_questions": [
                "What evidence supports improved patient outcomes?"
            ],
        }
    )

    def fake_run_research(
        state,
        llm_service,
        on_selection_recovery=None,
    ):
        assert on_selection_recovery is not None
        on_selection_recovery(
            "What evidence supports improved patient outcomes?"
        )
        return updated_state

    monkeypatch.setattr(
        "src.workflow.run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        "src.workflow.analyze_research",
        lambda state, llm_service: Analysis(
            findings=[],
            conflicts=[],
            gaps=[],
        ),
    )

    events: list[dict] = []

    def recorder(*args, **kwargs) -> None:
        events.append(kwargs)

    _research_node(state, recorder)

    recovery_events = [
        event
        for event in events
        if event["event_type"] == "search_selection_recovered"
    ]

    assert len(recovery_events) == 1
    assert recovery_events[0]["stage"] == "research"
    assert (
        recovery_events[0]["message"]
        == (
            'Search-candidate selection for '
            '"What evidence supports improved patient outcomes?" '
            "required local recovery from malformed structured output; "
            "recovered indices passed deterministic validation."
        )
    )
