from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

import httpx
from groq import BadRequestError

from src.agents.orchestrator import (
    CorrectionIssueMappingError,
    CorrectionQueryValidationError,
    _extract_recovered_correction_queries,
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

    queries, recovered = generate_correction_queries(
        "AI in healthcare",
        Q1,
        ("[RESEARCH] issue",),
        llm,
    )

    assert queries == ["current evidence patient outcomes"]
    assert recovered is False
    assert llm.invoke_structured.call_count == 2


def test_generate_correction_queries_fails_after_retry_budget() -> None:
    llm = Mock()
    llm.invoke_structured.return_value = CorrectionQueryPlan(queries=[])

    with pytest.raises(CorrectionQueryValidationError):
        generate_correction_queries("AI in healthcare", Q1, ("[RESEARCH] issue",), llm)

    assert llm.invoke_structured.call_count == 3


def _output_parse_failed_error(failed_generation: str) -> BadRequestError:
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
    )
    return BadRequestError(
        "structured output parsing failed",
        response=response,
        body={
            "error": {
                "message": "Parsing failed.",
                "type": "invalid_request_error",
                "code": "output_parse_failed",
                "failed_generation": failed_generation,
            }
        },
    )


def test_recover_correction_queries_from_quoted_failed_generation() -> None:
    error = _output_parse_failed_error(
        'Need queries: "GitHub Copilot productivity study" '
        '"ChatGPT code quality defect rate" '
        '"empirical evaluation of AI coding assistants" '
        '"AI pair programming defect metrics".'
    )

    recovered = _extract_recovered_correction_queries(error)

    assert recovered == [
        "GitHub Copilot productivity study",
        "ChatGPT code quality defect rate",
        "empirical evaluation of AI coding assistants",
    ]


def test_recover_correction_queries_from_bulleted_failed_generation() -> None:
    error = _output_parse_failed_error(
        "Need queries:\n"
        "- GitHub Copilot productivity study\n"
        "- empirical evaluation of AI coding assistants\n"
        "- Copilot impact on bug rates"
    )

    recovered = _extract_recovered_correction_queries(error)

    assert recovered == [
        "GitHub Copilot productivity study",
        "empirical evaluation of AI coding assistants",
        "Copilot impact on bug rates",
    ]


def test_unstructured_failed_generation_is_not_recovered() -> None:
    error = _output_parse_failed_error(
        "Need queries: recent industry reports, studies 2024-2026 on costs vs gains."
    )

    assert _extract_recovered_correction_queries(error) is None


def test_non_output_parse_bad_request_is_not_recovered() -> None:
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
    )
    error = BadRequestError(
        "malformed request",
        response=response,
        body={
            "error": {
                "message": "Malformed request.",
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        },
    )

    assert _extract_recovered_correction_queries(error) is None

    llm = Mock()
    llm.invoke_structured.side_effect = error

    with pytest.raises(BadRequestError):
        generate_correction_queries(
            "AI in software engineering",
            Q1,
            ("[RESEARCH] issue",),
            llm,
        )

    assert llm.invoke_structured.call_count == 1


def test_generate_correction_queries_recovers_without_second_llm_call() -> None:
    llm = Mock()
    llm.invoke_structured.side_effect = _output_parse_failed_error(
        'Need queries: "GitHub Copilot productivity study" '
        '"empirical evaluation of AI coding assistants".'
    )

    queries, recovered = generate_correction_queries(
        "AI in software engineering",
        Q1,
        ("[RESEARCH] issue",),
        llm,
    )

    assert queries == [
        "GitHub Copilot productivity study",
        "empirical evaluation of AI coding assistants",
    ]
    assert recovered is True
    assert llm.invoke_structured.call_count == 1


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

    updated, recovered = run_research_correction(
        state,
        [f'[RESEARCH] Sub-question: "{Q1}" needs current evidence.'],
        llm,
    )

    assert recovered is False
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

    updated, recovered = run_research_correction(
        state,
        [f'[RESEARCH] Sub-question 1 — "{Q1}" needs more evidence.'],
        llm,
    )

    assert recovered is False
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


def test_research_correction_continues_after_local_query_recovery(
    monkeypatch,
) -> None:
    state = _state()
    llm = Mock()

    llm.invoke_structured.side_effect = _output_parse_failed_error(
        'Need queries: "latest patient outcome evidence".'
    )

    candidates = [
        SearchResult(
            title="Recovered candidate",
            url="https://recovered.example/1",
            snippet="evidence",
        )
    ]

    monkeypatch.setattr(
        "src.agents.orchestrator.web_search",
        lambda query, max_results: candidates,
    )
    monkeypatch.setattr(
        "src.agents.research_agent.fetch_page_content",
        lambda url: "usable evidence " * 40,
    )

    updated, recovered = run_research_correction(
        state,
        [f'[RESEARCH] Sub-question: "{Q1}" needs current evidence.'],
        llm,
    )

    assert recovered is True
    assert updated.sources[-1].url == "https://recovered.example/1"
    assert updated.sources[-1].search_queries == [Q1]
    assert llm.invoke_structured.call_count == 1
