"""API-layer exception types and centralized error mapping helpers."""

from __future__ import annotations

from typing import Any

from src.core.persistence import PersistenceError


class APIError(RuntimeError):
    """Base class for expected API failures."""

    status_code = 500
    error = "InternalServerError"

    def __init__(self, message: str, *, detail: str | None = None, report_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.report_id = report_id


class ReportNotFoundError(APIError):
    status_code = 404
    error = "ReportNotFound"

    def __init__(self, report_id: str) -> None:
        super().__init__(
            "Report was not found.",
            detail="No completed report exists for the requested report_id.",
            report_id=report_id,
        )


class PersistenceAPIError(APIError):
    status_code = 500
    error = "PersistenceError"

    def __init__(self, message: str = "Unable to persist research data.", *, report_id: str | None = None) -> None:
        super().__init__(message, report_id=report_id)


class UpstreamDependencyError(APIError):
    status_code = 502
    error = "UpstreamDependencyError"

    def __init__(self, message: str = "An upstream dependency prevented the research workflow from completing.", *, detail: str | None = None, report_id: str | None = None) -> None:
        super().__init__(message, detail=detail, report_id=report_id)


class ResearchExecutionError(APIError):
    status_code = 500
    error = "ResearchExecutionError"

    def __init__(self, message: str, *, detail: str | None = None, report_id: str | None = None) -> None:
        super().__init__(message, detail=detail, report_id=report_id)


class ExecutionConflictError(APIError):
    status_code = 409
    error = "ConflictError"

    def __init__(self, message: str = "The research execution cannot be started in its current state.", *, detail: str | None = None, report_id: str | None = None) -> None:
        super().__init__(message, detail=detail, report_id=report_id)


def error_payload(exc: APIError) -> dict[str, Any]:
    return {
        "error": exc.error,
        "message": exc.message,
        "detail": exc.detail,
        "report_id": exc.report_id,
    }
