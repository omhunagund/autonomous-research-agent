from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from frontend.api_client import ResearchAPIClient
from frontend.models import APIClientError, ExecutionStatus, QualityLevel


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self):
        return self._payload


def test_initialize_research_uses_expected_contract(monkeypatch) -> None:
    calls = []

    def fake_request(method, url, *, json=None, timeout=None):
        calls.append((method, url, json, timeout))
        return FakeResponse(200, {"report_id": "abc", "status": "running"})

    monkeypatch.setattr(httpx, "request", fake_request)

    client = ResearchAPIClient("http://localhost:8000", timeout=7)
    result = client.initialize_research("test topic")

    assert result.report_id == "abc"
    assert result.status is ExecutionStatus.RUNNING
    assert calls == [
        ("POST", "http://localhost:8000/research/init", {"topic": "test topic"}, 7)
    ]


def test_execute_and_trace_are_typed(monkeypatch) -> None:
    responses = iter(
        [
            FakeResponse(200, {"report_id": "abc", "status": "completed"}),
            FakeResponse(
                200,
                {
                    "report_id": "abc",
                    "status": "completed",
                    "events": [
                        {
                            "event_id": "event-1",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "stage": "orchestrator",
                            "event_type": "execution_completed",
                            "message": "Done.",
                            "attempt": 1,
                        }
                    ],
                },
            ),
        ]
    )

    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: next(responses))

    client = ResearchAPIClient("http://localhost:8000")
    execute = client.execute_research("abc")
    trace = client.get_trace("abc")

    assert execute.status is ExecutionStatus.COMPLETED
    assert trace.status is ExecutionStatus.COMPLETED
    assert trace.events[0].event_type == "execution_completed"


def test_history_and_active_are_parsed(monkeypatch) -> None:
    responses = iter(
        [
            FakeResponse(
                200,
                {
                    "reports": [
                        {
                            "report_id": "abc",
                            "topic": "AI topic",
                            "created_at": "2026-09-06T10:00:00+00:00",
                            "quality": "high",
                        }
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "executions": [
                        {
                            "report_id": "def",
                            "topic": "Running topic",
                            "status": "running",
                            "attempt": 2,
                        }
                    ]
                },
            ),
        ]
    )

    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: next(responses))

    client = ResearchAPIClient("http://localhost:8000")
    history = client.get_history()
    active = client.get_active_research()

    assert history[0].quality is QualityLevel.HIGH
    assert active[0].status is ExecutionStatus.RUNNING
    assert active[0].attempt == 2


def test_structured_api_error_is_exposed(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "request",
        lambda *args, **kwargs: FakeResponse(
            409,
            {
                "error": "ExecutionConflict",
                "message": "Research is already running.",
                "detail": "Only one execution is allowed per report_id.",
                "report_id": "abc",
            },
        ),
    )

    client = ResearchAPIClient("http://localhost:8000")

    with pytest.raises(APIClientError) as exc_info:
        client.execute_research("abc")

    assert exc_info.value.status_code == 409
    assert exc_info.value.error == "ExecutionConflict"
    assert exc_info.value.report_id == "abc"
