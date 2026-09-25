from datetime import datetime, timezone
from unittest.mock import Mock

import httpx
import pytest
from groq import BadRequestError

from src.agents.research_agent import (
    ResearchPlanningError,
    SearchSelectionError,
    _attempt_candidates,
    _extract_recovered_selection,
    _validate_selection,
    _validate_sub_questions,
    generate_sub_questions,
    run_research,
    select_search_candidates,
)
from src.models.schemas import (
    ResearchState,
    SearchResult,
    SearchSelection,
    Source,
    SubQuestionPlan,
)
from src.tools.page_fetcher import PageFetchError


def _search_results(count: int = 5) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"Source {i}",
            url=f"https://example.com/{i}",
            snippet=f"Useful evidence snippet {i}",
        )
        for i in range(1, count + 1)
    ]


def _output_parse_failed_error(
    failed_generation: str,
    code: str = "output_parse_failed",
) -> BadRequestError:
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
                "code": code,
                "failed_generation": failed_generation,
            }
        },
    )


def _state() -> ResearchState:
    return ResearchState(
        user_topic="Artificial intelligence in healthcare",
        memory_context=None,
        sub_questions=[],
        sources=[],
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


def test_sub_question_validation_accepts_four_distinct_questions() -> None:
    plan = SubQuestionPlan(
        sub_questions=[
            "What are the main clinical uses of artificial intelligence?",
            "What evidence supports improved patient outcomes?",
            "What are the major risks and limitations of these systems?",
            "Which regulatory developments currently affect healthcare deployment?",
        ]
    )
    assert _validate_sub_questions(plan) == (True, "")


def test_sub_question_validation_rejects_duplicate_after_normalization() -> None:
    plan = SubQuestionPlan(
        sub_questions=[
            "What are the main uses of artificial intelligence today?",
            "What are the main uses of artificial intelligence today!",
            "What evidence supports improved patient outcomes?",
            "What are the major deployment risks and limitations today?",
        ]
    )
    valid, reason = _validate_sub_questions(plan)
    assert not valid
    assert "duplicate" in reason


def test_generate_sub_questions_uses_feedback_retry() -> None:
    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = [
        SubQuestionPlan(
            sub_questions=[
                "Too short",
                "Too short again",
                "Another question with enough words",
                "Yet another sufficiently detailed question",
            ]
        ),
        SubQuestionPlan(
            sub_questions=[
                "What are the main uses of artificial intelligence today?",
                "What evidence supports improved patient outcomes?",
                "What are the major risks and limitations for deployment?",
                "Which regulatory developments currently affect deployment?",
            ]
        ),
    ]

    result = generate_sub_questions(
        "Artificial intelligence in healthcare",
        mock_llm,
    )

    assert len(result) == 4
    assert mock_llm.invoke_structured.call_count == 2


def test_generate_sub_questions_fails_after_retry_budget() -> None:
    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = SubQuestionPlan(
        sub_questions=["Too short"] * 4
    )

    with pytest.raises(ResearchPlanningError):
        generate_sub_questions("Artificial intelligence in healthcare", mock_llm)

    assert mock_llm.invoke_structured.call_count == 3


def test_selection_validation_requires_exactly_three_unique_indices() -> None:
    valid, reason = _validate_selection(
        SearchSelection(selected_indices=[1, 3, 5])
    )
    assert valid
    assert reason == ""

    valid, reason = _validate_selection(
        SearchSelection(selected_indices=[1, 1, 5])
    )
    assert not valid
    assert "unique" in reason


def test_select_search_candidates_uses_feedback_retry() -> None:
    candidates = _search_results()
    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = [
        SearchSelection(selected_indices=[1, 1, 2]),
        SearchSelection(selected_indices=[2, 4, 5]),
    ]

    selected = select_search_candidates(
        "AI in healthcare",
        "What evidence supports improved patient outcomes?",
        candidates,
        mock_llm,
    )

    assert [item.url for item in selected] == [
        "https://example.com/2",
        "https://example.com/4",
        "https://example.com/5",
    ]
    assert mock_llm.invoke_structured.call_count == 2


def test_recover_search_selection_from_malformed_failed_generation() -> None:
    error = _output_parse_failed_error(
        "Need 3 most relevant. Likely 1,4,3 maybe."
    )

    recovered = _extract_recovered_selection(error)

    assert recovered == [1, 4, 3]


def test_recover_search_selection_from_probably_failed_generation() -> None:
    error = _output_parse_failed_error(
        "Need 3 most relevant. Probably 2,4,5."
    )

    recovered = _extract_recovered_selection(error)

    assert recovered == [2, 4, 5]


def test_select_search_candidates_passes_schema_recovery_handler():
    candidates = _search_results()
    mock_llm = Mock()

    def invoke_structured(*args, **kwargs):
        recovery_handler = kwargs["recovery_handler"]

        error = _output_parse_failed_error(
            "Need 3 most relevant. Probably 2,4,5.",
            code="tool_use_failed",
        )

        return recovery_handler(error)

    mock_llm.invoke_structured.side_effect = invoke_structured

    selected = select_search_candidates(
        "AI in healthcare",
        "What evidence supports improved patient outcomes?",
        candidates,
        mock_llm,
    )

    assert [item.url for item in selected] == [
        "https://example.com/2",
        "https://example.com/4",
        "https://example.com/5",
    ]

    mock_llm.invoke_structured.assert_called_once()


def test_recover_search_selection_from_json_validate_failed_generation() -> None:
    error = _output_parse_failed_error(
        '{"selected_indices":[1,3,2]',
        code="json_validate_failed",
    )

    recovered = _extract_recovered_selection(error)

    assert recovered == [1, 3, 2]


def test_select_search_candidates_recovers_without_second_llm_call() -> None:
    candidates = _search_results()
    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = _output_parse_failed_error(
        "Need 3 most relevant. Likely 1,4,3 maybe."
    )

    selected = select_search_candidates(
        "AI in healthcare",
        "What evidence supports improved patient outcomes?",
        candidates,
        mock_llm,
    )

    assert [item.url for item in selected] == [
        "https://example.com/1",
        "https://example.com/4",
        "https://example.com/3",
    ]
    assert mock_llm.invoke_structured.call_count == 1


def test_select_search_candidates_recovers_json_validate_failure_without_second_llm_call() -> None:
    candidates = _search_results()
    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = _output_parse_failed_error(
        '{"selected_indices":[1,3,2]',
        code="json_validate_failed",
    )

    selected = select_search_candidates(
        "AI in healthcare",
        "What evidence supports improved patient outcomes?",
        candidates,
        mock_llm,
    )

    assert [item.url for item in selected] == [
        "https://example.com/1",
        "https://example.com/3",
        "https://example.com/2",
    ]
    assert mock_llm.invoke_structured.call_count == 1


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

    mock_llm = Mock()
    mock_llm.invoke_structured.side_effect = error

    with pytest.raises(BadRequestError):
        select_search_candidates(
            "AI in healthcare",
            "What evidence supports improved patient outcomes?",
            _search_results(),
            mock_llm,
        )

    assert mock_llm.invoke_structured.call_count == 1


def test_select_search_candidates_fails_after_retry_budget() -> None:
    candidates = _search_results()
    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = SearchSelection(
        selected_indices=[1, 1, 2]
    )

    with pytest.raises(SearchSelectionError):
        select_search_candidates(
            "AI in healthcare",
            "What evidence supports improved patient outcomes?",
            candidates,
            mock_llm,
        )

    assert mock_llm.invoke_structured.call_count == 3


def test_attempt_candidates_falls_through() -> None:
    candidates = _search_results()
    source_by_url: dict[str, Source] = {}
    ordered_sources: list[Source] = []
    citation_counter = [1]

    def fake_fetch(url: str) -> str:
        if url.endswith("/2"):
            raise PageFetchError("blocked")
        return ("usable evidence " * 80).strip()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("src.agents.research_agent.fetch_page_content", fake_fetch)
    try:
        limitation = _attempt_candidates(
            sub_question="What evidence supports improved patient outcomes?",
            candidates=candidates,
            source_by_url=source_by_url,
            ordered_sources=ordered_sources,
            citation_counter=citation_counter,
        )
    finally:
        monkeypatch.undo()

    assert limitation.usable_source_count == 4
    assert limitation.failed_candidate_count == 1
    assert [source.citation_id for source in ordered_sources] == [1, 2, 3, 4]


def test_attempt_candidates_counts_existing_global_source_for_new_question() -> None:
    existing = Source(
        citation_id=1,
        title="Shared source",
        url="https://example.com/shared",
        retrieved_at=datetime.now(timezone.utc),
        snippet="shared evidence",
        search_queries=["first question"],
        content="usable evidence",
    )
    source_by_url = {existing.url: existing}
    ordered_sources = [existing]
    citation_counter = [2]

    candidates = [
        SearchResult(
            title="Shared source",
            url="https://example.com/shared",
            snippet="shared evidence",
        )
    ]

    limitation = _attempt_candidates(
        sub_question="What evidence supports improved patient outcomes?",
        candidates=candidates,
        source_by_url=source_by_url,
        ordered_sources=ordered_sources,
        citation_counter=citation_counter,
    )

    assert limitation.usable_source_count == 1
    assert limitation.failed_candidate_count == 0
    assert existing.search_queries == [
        "first question",
        "What evidence supports improved patient outcomes?",
    ]
    assert len(ordered_sources) == 1


def test_research_does_not_call_selection_llm_for_four_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = SubQuestionPlan(
        sub_questions=[
            "What are the main clinical uses of artificial intelligence?",
            "What evidence supports improved patient outcomes?",
            "What are the major risks and limitations for deployment?",
            "Which regulatory developments currently affect deployment?",
        ]
    )

    four_results = _search_results(4)

    monkeypatch.setattr(
        "src.agents.research_agent.web_search",
        lambda query, max_results=5: four_results,
    )
    monkeypatch.setattr(
        "src.agents.research_agent.fetch_page_content",
        lambda url: ("usable evidence " * 80).strip(),
    )

    state = run_research(_state(), mock_llm)

    assert len(state.sub_questions) == 4
    assert len(state.sources) == 4
    assert all(
        "fewer than the expected 5" in limitation.description
        for limitation in state.research_limitations
    )
    # One structured call generated the plan; no SearchSelection call occurred.
    assert mock_llm.invoke_structured.call_count == 1


def test_research_does_not_call_selection_llm_for_three_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = SubQuestionPlan(
        sub_questions=[
            "What are the main clinical uses of artificial intelligence?",
            "What evidence supports improved patient outcomes?",
            "What are the major risks and limitations for deployment?",
            "Which regulatory developments currently affect deployment?",
        ]
    )

    three_results = _search_results(3)

    monkeypatch.setattr(
        "src.agents.research_agent.web_search",
        lambda query, max_results=5: three_results,
    )
    monkeypatch.setattr(
        "src.agents.research_agent.fetch_page_content",
        lambda url: ("usable evidence " * 80).strip(),
    )

    state = run_research(_state(), mock_llm)

    assert len(state.sources) == 3
    assert mock_llm.invoke_structured.call_count == 1
    assert all(item.usable_source_count == 3 for item in state.research_limitations)


def test_research_records_zero_result_limitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_llm = Mock()
    mock_llm.invoke_structured.return_value = SubQuestionPlan(
        sub_questions=[
            "What are the main clinical uses of artificial intelligence?",
            "What evidence supports improved patient outcomes?",
            "What are the major risks and limitations for deployment?",
            "Which regulatory developments currently affect deployment?",
        ]
    )

    monkeypatch.setattr(
        "src.agents.research_agent.web_search",
        lambda query, max_results=5: [],
    )

    state = run_research(_state(), mock_llm)

    assert len(state.sources) == 0
    assert len(state.research_limitations) == 4
    assert all(item.usable_source_count == 0 for item in state.research_limitations)
    assert all("no candidates" in item.description for item in state.research_limitations)
