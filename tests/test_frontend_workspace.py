from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from frontend.app import (
    _citation_links_html,
    _current_stage,
    _event_completed,
    _jump_to_latest_html,
    _source_lookup,
    _trace_event_signature,
)
from frontend.models import TraceEvent


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
