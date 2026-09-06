"""FastAPI route definitions for research, report retrieval, trace, and health."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from src.api.errors import APIError, PersistenceAPIError, ReportNotFoundError, UpstreamDependencyError
from src.api.schemas import (
    ExecutionStatus,
    HistoryItem,
    HistoryResponse,
    ResearchRequest,
    ResearchResponse,
    TraceEvent,
    TraceEventType,
    TraceResponse,
)
from src.core.execution import ResearchExecutionError, ResearchExecutionService
from src.core.llm import LLMConfigurationError, LLMInvocationError
from src.core.persistence import PersistenceError
from src.tools.page_fetcher import PageFetchError
from src.tools.web_search import SearchError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_execution_service(request: Request) -> ResearchExecutionService:
    """Return the application-scoped execution service."""
    service = getattr(request.app.state, "execution_service", None)
    if service is None:
        raise RuntimeError("ExecutionService has not been initialized")
    return service


def _handle_execution_error(exc: ResearchExecutionError) -> APIError:
    cause = exc.cause
    if isinstance(cause, (LLMInvocationError, SearchError, PageFetchError)):
        return UpstreamDependencyError(report_id=exc.report_id)
    if isinstance(cause, LLMConfigurationError):
        return APIError(
            "Research service is not correctly configured.",
            detail="Required application configuration is missing or invalid.",
            report_id=exc.report_id,
        )
    if isinstance(cause, PersistenceError):
        return PersistenceAPIError(report_id=exc.report_id)
    return APIError(
        "An unexpected internal error occurred.",
        detail="The research request could not be completed.",
        report_id=exc.report_id,
    )


@router.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
)
def create_research(
    payload: ResearchRequest,
    service: ResearchExecutionService = Depends(get_execution_service),
) -> ResearchResponse:
    try:
        execution = service.run(payload.topic)
    except ResearchExecutionError as exc:
        raise _handle_execution_error(exc) from exc
    except PersistenceError as exc:
        logger.exception("Persistence failure while starting research")
        raise PersistenceAPIError() from exc
    except Exception as exc:
        logger.exception("Unexpected research execution failure")
        raise APIError(
            "An unexpected internal error occurred.",
            detail="The research request could not be completed.",
        ) from exc

    if execution.report.report is None:
        raise APIError(
            "An unexpected internal error occurred.",
            detail="The research execution completed without producing a final report.",
            report_id=execution.report_id,
        )

    return ResearchResponse(report_id=execution.report_id, report=execution.report.report)


@router.get(
    "/reports/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
)
def get_history(
    service: ResearchExecutionService = Depends(get_execution_service),
) -> HistoryResponse:
    try:
        reports = service.persistence.list_reports(limit=20)
    except PersistenceError as exc:
        logger.exception("Persistence failure while reading report history")
        raise PersistenceAPIError() from exc

    return HistoryResponse(
        reports=[
            HistoryItem(
                report_id=report.report_id,
                topic=report.topic,
                created_at=report.created_at,
                quality=report.quality,
            )
            for report in reports
        ]
    )



@router.get(
    "/reports/{report_id}",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
)
def get_report(
    report_id: UUID,
    service: ResearchExecutionService = Depends(get_execution_service),
) -> ResearchResponse:
    report_id_str = str(report_id)
    try:
        report = service.persistence.get_report(report_id_str)
    except PersistenceError as exc:
        logger.exception("Persistence failure while reading report %s", report_id_str)
        raise PersistenceAPIError(report_id=report_id_str) from exc

    if report is None:
        raise ReportNotFoundError(report_id_str)
    return ResearchResponse(report_id=report.report_id, report=report.report)


@router.get(
    "/research/{report_id}/trace",
    response_model=TraceResponse,
    status_code=status.HTTP_200_OK,
)
def get_trace(
    report_id: UUID,
    service: ResearchExecutionService = Depends(get_execution_service),
) -> TraceResponse:
    report_id_str = str(report_id)
    try:
        execution_status = service.persistence.get_execution_status(report_id_str)
        if execution_status is None:
            raise ReportNotFoundError(report_id_str)
        events = service.persistence.get_trace(report_id_str)
    except ReportNotFoundError:
        raise
    except PersistenceError as exc:
        logger.exception("Persistence failure while reading trace %s", report_id_str)
        raise PersistenceAPIError(report_id=report_id_str) from exc

    try:
        api_status = ExecutionStatus(execution_status)
    except ValueError as exc:
        raise APIError(
            "An unexpected internal error occurred.",
            detail="The persisted execution status is invalid.",
            report_id=report_id_str,
        ) from exc

    return TraceResponse(
        report_id=report_id_str,
        status=api_status,
        events=[
            TraceEvent(
                event_id=event.event_id,
                timestamp=event.timestamp,
                stage=event.stage,
                event_type=TraceEventType(event.event_type),
                message=event.message,
                attempt=event.attempt,
            )
            for event in events
        ],
    )


@router.get("/health", response_model=dict[str, str], status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    return {"status": "ok"}
