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
    _render_trace,
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

def test_trace_events_are_rendered_grouped_by_attempt(monkeypatch) -> None:
    rendered: list[str] = []

    monkeypatch.setattr(
        "frontend.app._jump_to_latest_html",
        lambda: "",
    )
    monkeypatch.setattr(
        "frontend.app._trace_event_signature",
        lambda events: tuple(event.event_id for event in events),
    )
    monkeypatch.setattr(
        "frontend.app.st",
        SimpleNamespace(
            subheader=lambda value: rendered.append(f"SUBHEADER:{value}"),
            caption=lambda value: rendered.append(f"CAPTION:{value}"),
            markdown=lambda value, **kwargs: rendered.append(value),
            session_state={
                "trace_seen_event_ids": tuple(),
                "trace_has_new_events": False,
            },
        ),
    )

    events = [
        _event(
            "research",
            event_type="research_started",
            attempt=1,
        ),
        _event(
            "analysis",
            event_type="analysis_completed",
            attempt=1,
        ),
        _event(
            "research",
            event_type="research_started",
            attempt=2,
        ),
        _event(
            "critic",
            event_type="critic_review_completed",
            attempt=2,
        ),
    ]

    _render_trace(events)

    output = "\n".join(rendered)

    first_attempt = output.index("Attempt 1")
    second_attempt = output.index("Attempt 2")

    assert first_attempt < second_attempt

    attempt_1_output = output[first_attempt:second_attempt]
    attempt_2_output = output[second_attempt:]

    assert "Research" in attempt_1_output
    assert "Analysis" in attempt_1_output
    assert "Attempt 1" in attempt_1_output

    assert "Research" in attempt_2_output
    assert "Critic" in attempt_2_output
    assert "Attempt 2" in attempt_2_output


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

def test_terminal_trace_requests_full_app_rerun(monkeypatch) -> None:
    import frontend.app as app

    rerun_calls = []

    class FakeClient:
        def get_active_research(self):
            return []

        def get_history(self):
            return []

    monkeypatch.setattr(
        "frontend.app.st.rerun",
        lambda: rerun_calls.append(True),
    )
    monkeypatch.setattr(
        "frontend.app._refresh_active",
        lambda client: None,
    )
    monkeypatch.setattr(
        "frontend.app._refresh_history",
        lambda client: None,
    )

    app._handle_terminal_trace_refresh(
        FakeClient(),
        app.ExecutionStatus.COMPLETED,
    )

    assert rerun_calls == [True]


def test_terminal_failure_requests_full_app_rerun(monkeypatch) -> None:
    import frontend.app as app

    rerun_calls = []

    class FakeClient:
        def get_active_research(self):
            return []

    monkeypatch.setattr(
        "frontend.app.st.rerun",
        lambda: rerun_calls.append(True),
    )
    monkeypatch.setattr(
        "frontend.app._refresh_active",
        lambda client: None,
    )

    app._handle_terminal_trace_refresh(
        FakeClient(),
        app.ExecutionStatus.FAILED,
    )

    assert rerun_calls == [True]

def test_streamlit_status_state_distinguishes_failed_execution(monkeypatch) -> None:
    import frontend.app as app

    captured = {}

    def fake_status(label, state=None):
        captured["label"] = label
        captured["state"] = state

    monkeypatch.setattr("frontend.app.st.status", fake_status)
    monkeypatch.setattr(
        "frontend.app._render_trace",
        lambda events: None,
    )
    monkeypatch.setattr(
        "frontend.app._complete_live_workspace",
        lambda client, report_id: None,
    )

    app.st.session_state = {
        "selected_report_id": "report-1",
        "selected_topic": "Failure test",
        "current_trace": type(
            "Trace",
            (),
            {
                "report_id": "report-1",
                "status": app.ExecutionStatus.FAILED,
                "events": [],
            },
        )(),
    }

    app._render_live_workspace(object())

    assert captured["state"] == "error"

def test_reference_url_allows_http_and_https(monkeypatch) -> None:
    import frontend.app as app

    monkeypatch.setattr(
        "frontend.app.st.markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    rendered = []

    report = {
        "executive_summary": "",
        "key_findings": [],
        "supporting_evidence": [],
        "gaps": [],
        "gap_explanations": [],
        "conflicts": [],
        "conflict_explanations": [],
        "references": [
            {
                "citation_id": 1,
                "title": "HTTP source",
                "url": "http://example.com/source",
            },
            {
                "citation_id": 2,
                "title": "HTTPS source",
                "url": "https://example.com/source",
            },
        ],
    }

    app._render_report(report)

    joined = "\n".join(rendered)

    assert 'href="http://example.com/source"' in joined
    assert 'href="https://example.com/source"' in joined


def test_reference_url_rejects_non_http_scheme(monkeypatch) -> None:
    import frontend.app as app

    monkeypatch.setattr(
        "frontend.app.st.markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    rendered = []

    report = {
        "executive_summary": "",
        "key_findings": [],
        "supporting_evidence": [],
        "gaps": [],
        "gap_explanations": [],
        "conflicts": [],
        "conflict_explanations": [],
        "references": [
            {
                "citation_id": 1,
                "title": "Unsafe source",
                "url": "javascript:alert(1)",
            },
        ],
    }

    app._render_report(report)

    joined = "\n".join(rendered)

    assert 'href="javascript:alert(1)"' not in joined
    assert "Unsafe source" in joined

def test_live_workspace_new_research_resets_workspace_without_clearing_active(monkeypatch) -> None:
    import frontend.app as app

    app.st.session_state = {
        "workspace_mode": "live",
        "selected_report_id": "report-1",
        "selected_topic": "Running research",
        "selected_quality": app.QualityLevel.HIGH,
        "current_report": {"topic": "Running research"},
        "current_trace": object(),
        "report_error": "old error",
        "execution_error": "old execution error",
        "polling_active": True,
        "active_research": [
            type(
                "Active",
                (),
                {
                    "report_id": "report-1",
                    "topic": "Running research",
                    "status": app.ExecutionStatus.RUNNING,
                    "attempt": 1,
                },
            )()
        ],
    }

    rerun_calls = []

    monkeypatch.setattr(
        "frontend.app.st.button",
        lambda label, **kwargs: label == "New Research",
    )
    monkeypatch.setattr(
        "frontend.app.st.rerun",
        lambda: rerun_calls.append(True),
    )

    app._render_live_workspace_controls()

    assert app.st.session_state["workspace_mode"] == "landing"
    assert app.st.session_state["selected_report_id"] is None
    assert app.st.session_state["selected_topic"] is None
    assert app.st.session_state["selected_quality"] is None
    assert app.st.session_state["current_report"] is None
    assert app.st.session_state["current_trace"] is None
    assert app.st.session_state["report_error"] is None
    assert app.st.session_state["execution_error"] is None
    assert app.st.session_state["polling_active"] is False
    assert len(app.st.session_state["active_research"]) == 1
    assert rerun_calls == [True]

def test_active_research_selected_state_is_distinguished(monkeypatch) -> None:
    import frontend.app as app

    rendered_labels = []

    class FakeSidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeItem:
        def __init__(self, report_id: str, topic: str, attempt: int) -> None:
            self.report_id = report_id
            self.topic = topic
            self.attempt = attempt
            self.status = app.ExecutionStatus.RUNNING

    active_items = [
        FakeItem("report-1", "First research", 1),
        FakeItem("report-2", "Second research", 2),
    ]

    app.st.session_state = {
        "active_research": active_items,
        "active_error": None,
        "history_error": None,
        "history": [],
        "selected_report_id": "report-2",
        "workspace_mode": "live",
    }

    monkeypatch.setattr(
        "frontend.app.st.sidebar",
        FakeSidebar(),
    )
    monkeypatch.setattr(
        "frontend.app.st.subheader",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "frontend.app.st.markdown",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "frontend.app.st.caption",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "frontend.app.st.error",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "frontend.app.st.button",
        lambda label, **kwargs: (
            rendered_labels.append(label) or False
        ),
    )

    app._render_sidebar(object())

    assert rendered_labels[0] == "First research  \nRunning · Attempt 1"
    assert rendered_labels[1] == "• Second research  \nRunning · Attempt 2"
