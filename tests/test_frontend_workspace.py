from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from frontend.app import _current_stage, _event_completed
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
