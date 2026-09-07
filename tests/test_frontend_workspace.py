from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from frontend.app import (
    _citation_links_html,
    _current_stage,
    _event_completed,
    _history_topic_label,
    _jump_to_latest_html,
    _source_lookup,
    _trace_event_signature,
)
from frontend.models import TraceEvent

from frontend.app import _refresh_after_terminal_trace
from frontend.models import ExecutionStatus


def _event(stage: str, event_type: str, attempt: int = 1) -> TraceEvent:
    return TraceEvent(
        event_id=f"{stage}-{event_type}",
        timestamp=datetime.now(timezone.utc),
        stage=stage,
        event_type=event_type,
        message="event",
        attempt=attempt,
    )


def test_current_stage_uses_latest_workflow_stage() -> None:
    events = [
        _event("research", "research_completed"),
        _event("analysis", "analysis_completed"),
        _event("writing", "writing_started"),
    ]
    assert _current_stage(events) == "writing"


def test_current_stage_ignores_orchestrator_events() -> None:
    events = [
        _event("orchestrator", "memory_matches_found"),
        _event("research", "research_started"),
        _event("orchestrator", "execution_completed"),
    ]
    assert _current_stage(events) == "research"


@pytest.mark.parametrize(
    ("stage", "event_type", "expected"),
    [
        ("research", "research_completed", True),
        ("research", "research_correction_completed", True),
        ("analysis", "analysis_completed", True),
        ("writing", "writing_completed", True),
        ("writing", "writing_correction_completed", True),
        ("critic", "critic_review_completed", True),
        ("research", "research_started", False),
        ("analysis", "writing_completed", False),
    ],
)
def test_event_completed(stage: str, event_type: str, expected: bool) -> None:
    assert _event_completed([_event(stage, event_type)], stage) is expected


def test_citation_links_render_only_approved_sources() -> None:
    rendered = _citation_links_html(
        "Claim [1] with [2] and [99] plus <unsafe>",
        {1, 2},
    )

    assert 'href="#source-1"' in rendered
    assert 'href="#source-2"' in rendered
    assert "[99]" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert 'href="#source-99"' not in rendered


def test_source_lookup_indexes_citation_ids() -> None:
    sources = [
        {"citation_id": 2, "title": "Second", "url": "https://example.com/2"},
        {"citation_id": 1, "title": "First", "url": "https://example.com/1"},
    ]

    lookup = _source_lookup(sources)

    assert list(lookup) == [2, 1]
    assert lookup[1]["title"] == "First"


def test_trace_event_signature_follows_event_order() -> None:
    events = [
        _event("research", "research_started"),
        _event("analysis", "analysis_completed"),
    ]
    assert _trace_event_signature(events) == tuple(event.event_id for event in events)


def test_jump_to_latest_uses_dependency_free_anchor() -> None:
    rendered = _jump_to_latest_html()

    assert 'href="#trace-latest"' in rendered
    assert 'role="button"' not in rendered
    assert "Jump to latest" in rendered

def test_history_topic_label_keeps_short_topic_unchanged() -> None:
    topic = "AI in healthcare"

    assert _history_topic_label(topic) == topic


def test_history_topic_label_truncates_long_topic_with_ellipsis() -> None:
    topic = (
        "How is generative AI changing software engineering "
        "productivity in modern development teams?"
    )

    result = _history_topic_label(topic)

    assert result.endswith("…")
    assert len(result) <= 42
    assert result != topic

def test_terminal_completion_refreshes_active_and_history(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        "frontend.app._refresh_active",
        lambda client: calls.append("active"),
    )
    monkeypatch.setattr(
        "frontend.app._refresh_history",
        lambda client: calls.append("history"),
    )

    _refresh_after_terminal_trace(
        object(),
        ExecutionStatus.COMPLETED,
    )

    assert calls == ["active", "history"]


def test_terminal_failure_refreshes_active_but_not_history(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        "frontend.app._refresh_active",
        lambda client: calls.append("active"),
    )
    monkeypatch.setattr(
        "frontend.app._refresh_history",
        lambda client: calls.append("history"),
    )

    _refresh_after_terminal_trace(
        object(),
        ExecutionStatus.FAILED,
    )

    assert calls == ["active"]

def test_complete_live_workspace_reuses_loaded_report(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeClient:
        def get_report(self, report_id):
            calls["count"] += 1
            return {
                "report": {
                    "quality": "high",
                    "key_findings": [],
                    "supporting_evidence": [],
                    "gaps": [],
                    "gap_explanations": [],
                    "conflicts": [],
                    "conflict_explanations": [],
                    "references": [],
                    "executive_summary": "Summary.",
                }
            }

    rendered_reports = []

    monkeypatch.setattr(
        "frontend.app._render_report",
        lambda report: rendered_reports.append(report),
    )
    monkeypatch.setattr(
        "frontend.app.st.success",
        lambda *args, **kwargs: None,
    )

    import frontend.app as app

    app.st.session_state = {
        "polling_active": False,
        "current_report": None,
        "selected_quality": None,
        "current_trace": type(
            "Trace",
            (),
            {"events": []},
        )(),
    }

    app._complete_live_workspace(FakeClient(), "report-1")
    app._complete_live_workspace(FakeClient(), "report-1")

    assert calls["count"] == 1
    assert len(rendered_reports) == 2
    assert rendered_reports[0] == rendered_reports[1]


def test_complete_live_workspace_sets_report_error_when_initial_load_fails(monkeypatch) -> None:
    import frontend.app as app

    class FakeClient:
        def get_report(self, report_id):
            raise app.APIClientError("Report unavailable")

    app.st.session_state = {
        "polling_active": False,
        "current_report": None,
        "selected_quality": None,
        "current_trace": type(
            "Trace",
            (),
            {"events": []},
        )(),
    }

    monkeypatch.setattr(
        "frontend.app.st.warning",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "frontend.app.st.button",
        lambda *args, **kwargs: False,
    )

    app._complete_live_workspace(FakeClient(), "report-1")

    assert app.st.session_state["report_error"] == "Report unavailable"

def test_start_research_does_not_store_report_execution_future(monkeypatch) -> None:
    import frontend.app as app

    class FakeClient:
        def initialize_research(self, topic):
            return type("Init", (), {"report_id": "report-1"})()

    submit_calls = []

    monkeypatch.setattr(
        "frontend.app._submit_execution",
        lambda client, report_id: submit_calls.append(report_id),
    )
    monkeypatch.setattr(
        "frontend.app._refresh_active",
        lambda client, select_id=None: None,
    )

    app.st.session_state = {}

    app._start_research(FakeClient(), "Test concurrent research")

    assert submit_calls == ["report-1"]
    assert "execution_future" not in app.st.session_state
