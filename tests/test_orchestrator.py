from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.agents.orchestrator import (
    CorrectionIssueMappingError,
    CorrectionQueryValidationError,
    _validate_correction_queries,
    generate_correction_queries,
    group_research_issues,
    run_research_correction,
)
from src.models.schemas import (
    CorrectionQueryPlan,
    FinalReport,
    ResearchLimitation,
    ResearchState,
    SearchResult,
    Source,
)


Q1 = "What current evidence supports improved patient outcomes?"
Q2 = "What are the major deployment risks and limitations?"


def _source(i: int, queries=None) -> Source:
    return Source(
        citation_id=i,
        title=f"Source {i}",
        url=f"https://example.com/{i}",
        retrieved_at=datetime.now(timezone.utc),
        snippet="Evidence",
        search_queries=queries or [Q1],
        content="usable evidence " * 40,
    )


def _state() -> ResearchState:
    return ResearchState(
        user_topic="AI in healthcare",
        memory_context=None,
        sub_questions=[Q1, Q2],
        sources=[_source(1, [Q1]), _source(2, [Q1])],
        findings=[],
        conflicts=[],
        gaps=[],
        research_limitations=[
            ResearchLimitation(
                sub_question=Q1,
                usable_source_count=2,
                failed_candidate_count=1,
                description="limited",
            )
        ],
        draft=None,
        critique=None,
        retry_count=1,
        revision_target="research",
        final_report=None,
    )


def test_groups_multiple_research_issues_for_same_exact_subquestion() -> None:
    issues = [
        f'[RESEARCH] Sub-question 1 — "{Q1}" lacks current clinical evidence.',
        f'[RESEARCH] Sub-question: "{Q1}" needs a second independent source.',
        f'[RESEARCH] Sub-question 2 — "{Q2}" lacks deployment evidence.',
    ]

    groups = group_research_issues(issues, [Q1, Q2])

    assert len(groups) == 2
    assert groups[0].sub_question == Q1
    assert len(groups[0].issues) == 2
    assert groups[1].sub_question == Q2
    assert len(groups[1].issues) == 1


def test_research_issue_requires_exact_current_subquestion() -> None:
    with pytest.raises(CorrectionIssueMappingError):
        group_research_issues(
            ['[RESEARCH] Sub-question: "A different question" lacks evidence.'],
            [Q1, Q2],
        )


def test_research_issue_requires_quoted_subquestion() -> None:
    with pytest.raises(CorrectionIssueMappingError):
        group_research_issues(
            ["[RESEARCH] Sub-question 1 lacks current evidence."],
            [Q1, Q2],
        )


def test_correction_query_plan_requires_one_to_three_unique_queries() -> None:
    for count in (1, 2, 3):
        valid, reason = _validate_correction_queries(
            CorrectionQueryPlan(queries=[f"query {i}" for i in range(count)])
        )
        assert valid, reason

    valid, reason = _validate_correction_queries(CorrectionQueryPlan(queries=[]))
    assert not valid
    assert "1-3" in reason

    valid, reason = _validate_correction_queries(
        CorrectionQueryPlan(queries=["same", "same"])
    )
    assert not valid
    assert "duplicate" in reason


def test_generate_correction_queries_retries_deterministic_validation() -> None:
    llm = Mock()
    llm.invoke_structured.side_effect = [
        CorrectionQueryPlan(queries=[]),
        CorrectionQueryPlan(queries=["current evidence patient outcomes"]),
    ]

    queries = generate_correction_queries("AI in healthcare", Q1, ("[RESEARCH] issue",), llm)

    assert queries == ["current evidence patient outcomes"]
    assert llm.invoke_structured.call_count == 2


def test_generate_correction_queries_fails_after_retry_budget() -> None:
    llm = Mock()
    llm.invoke_structured.return_value = CorrectionQueryPlan(queries=[])

    with pytest.raises(CorrectionQueryValidationError):
        generate_correction_queries("AI in healthcare", Q1, ("[RESEARCH] issue",), llm)

    assert llm.invoke_structured.call_count == 3


def test_research_correction_reuses_top3_selection_for_five_results(monkeypatch) -> None:
    state = _state()
    llm = Mock()
    llm.invoke_structured.return_value = CorrectionQueryPlan(
        queries=["latest patient outcome evidence"]
    )

    candidates = [
        SearchResult(title=f"Candidate {i}", url=f"https://new.example/{i}", snippet="evidence")
        for i in range(1, 6)
    ]
    selected = [candidates[0], candidates[2], candidates[4]]

    monkeypatch.setattr(
        "src.agents.orchestrator.web_search",
        lambda query, max_results: candidates,
    )
    monkeypatch.setattr(
        "src.agents.orchestrator.select_search_candidates",
        lambda topic, sub_question, got, service: selected,
    )
    monkeypatch.setattr(
        "src.agents.orchestrator.fetch_page_content",
        lambda url: "usable evidence " * 40,
        raising=False,
    )
    monkeypatch.setattr(
        "src.agents.research_agent.fetch_page_content",
        lambda url: "usable evidence " * 40,
    )

    updated = run_research_correction(
        state,
        [f'[RESEARCH] Sub-question: "{Q1}" needs current evidence.'],
        llm,
    )

    assert len(updated.sources) == 7
    assert updated.sources[-1].citation_id == 7
    assert updated.research_limitations == []


def test_research_correction_preserves_existing_sources_and_removes_resolved_limitations(
    monkeypatch,
) -> None:
    state = _state()
    llm = Mock()
    llm.invoke_structured.return_value = CorrectionQueryPlan(
        queries=["new outcome evidence"]
    )
    candidates = [
        SearchResult(title=f"Candidate {i}", url=f"https://new.example/{i}", snippet="evidence")
        for i in range(1, 4)
    ]
    monkeypatch.setattr(
        "src.agents.orchestrator.web_search",
        lambda query, max_results: candidates,
    )
    monkeypatch.setattr(
        "src.agents.orchestrator.select_search_candidates",
        lambda topic, sub_question, got, service: got[:3],
    )
    monkeypatch.setattr(
        "src.agents.research_agent.fetch_page_content",
        lambda url: "usable evidence " * 40,
    )

    updated = run_research_correction(
        state,
        [f'[RESEARCH] Sub-question 1 — "{Q1}" needs more evidence.'],
        llm,
    )

    assert [source.citation_id for source in updated.sources[:2]] == [1, 2]
    assert len(updated.sources) == 5
    assert updated.research_limitations == []


def test_final_report_schema_exposes_analysis_and_unresolved_issue_fields() -> None:
    report = FinalReport(
        topic="AI",
        executive_summary="AI summary",
        key_findings=[],
        supporting_evidence=[],
        gaps=[],
        gap_explanations=[],
        conflicts=[],
        conflict_explanations=[],
        references=[],
    )

    assert report.analysis_issues == []
    assert report.unresolved_issues == []
