"""Typed frontend representations of the FastAPI API contract."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class HistoryItem:
    report_id: str
    topic: str
    created_at: datetime
    quality: QualityLevel


@dataclass(frozen=True)
class ActiveExecution:
    report_id: str
    topic: str
    status: ExecutionStatus
    attempt: int


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    timestamp: datetime
    stage: str
    event_type: str
    message: str
    attempt: int


@dataclass(frozen=True)
class TraceResponse:
    report_id: str
    status: ExecutionStatus
    events: list[TraceEvent]


@dataclass(frozen=True)
class ResearchInitResponse:
    report_id: str
    status: ExecutionStatus


@dataclass(frozen=True)
class ResearchExecuteResponse:
    report_id: str
    status: ExecutionStatus


class APIClientError(RuntimeError):
    """Expected HTTP/API failure raised by the frontend client."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error: str | None = None,
        report_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error
        self.report_id = report_id


def _parse_error(response: Any) -> APIClientError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    message = payload.get("message") or "The API request could not be completed."
    return APIClientError(
        message,
        status_code=response.status_code,
        error=payload.get("error"),
        report_id=payload.get("report_id"),
    )
