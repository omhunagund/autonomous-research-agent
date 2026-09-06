"""HTTP client for the Autonomous Research & Report Agent API."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx

from frontend.models import (
    APIClientError,
    ActiveExecution,
    ExecutionStatus,
    HistoryItem,
    QualityLevel,
    ResearchExecuteResponse,
    ResearchInitResponse,
    TraceEvent,
    TraceResponse,
    _parse_error,
)

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 15.0


def get_api_base_url() -> str:
    """Read the backend URL, allowing an environment override."""
    return os.getenv("RESEARCH_AGENT_API_URL", DEFAULT_API_BASE_URL).rstrip("/")


class ResearchAPIClient:
    """Small typed HTTP client used by the Streamlit application."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved_url = (base_url or get_api_base_url()).rstrip("/")
        if not resolved_url:
            raise ValueError("API base URL cannot be blank.")
        self.base_url = resolved_url
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(
                method,
                url,
                json=json,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise APIClientError("Unable to reach the research API.") from exc

        if response.is_error:
            raise _parse_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "The research API returned an invalid JSON response.",
                status_code=response.status_code,
            ) from exc

        if not isinstance(payload, dict):
            raise APIClientError(
                "The research API returned an unexpected response shape.",
                status_code=response.status_code,
            )

        return payload

    def initialize_research(self, topic: str) -> ResearchInitResponse:
        payload = self._request(
            "POST",
            "/research/init",
            json={"topic": topic},
        )
        return ResearchInitResponse(
            report_id=str(payload["report_id"]),
            status=ExecutionStatus(payload["status"]),
        )

    def execute_research(self, report_id: str) -> ResearchExecuteResponse:
        payload = self._request(
            "POST",
            f"/research/{report_id}/execute",
        )
        return ResearchExecuteResponse(
            report_id=str(payload["report_id"]),
            status=ExecutionStatus(payload["status"]),
        )

    def get_trace(self, report_id: str) -> TraceResponse:
        payload = self._request("GET", f"/research/{report_id}/trace")
        return TraceResponse(
            report_id=str(payload["report_id"]),
            status=ExecutionStatus(payload["status"]),
            events=[
                TraceEvent(
                    event_id=str(event["event_id"]),
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                    stage=str(event["stage"]),
                    event_type=str(event["event_type"]),
                    message=str(event["message"]),
                    attempt=int(event["attempt"]),
                )
                for event in payload["events"]
            ],
        )

    def get_history(self) -> list[HistoryItem]:
        payload = self._request("GET", "/reports/history")
        return [
            HistoryItem(
                report_id=str(item["report_id"]),
                topic=str(item["topic"]),
                created_at=datetime.fromisoformat(item["created_at"]),
                quality=QualityLevel(item["quality"]),
            )
            for item in payload["reports"]
        ]

    def get_active_research(self) -> list[ActiveExecution]:
        payload = self._request("GET", "/research/active")
        return [
            ActiveExecution(
                report_id=str(item["report_id"]),
                topic=str(item["topic"]),
                status=ExecutionStatus(item["status"]),
                attempt=int(item["attempt"]),
            )
            for item in payload["executions"]
        ]

    def get_report(self, report_id: str) -> dict[str, Any]:
        return self._request("GET", f"/reports/{report_id}")

    def health(self) -> dict[str, str]:
        return self._request("GET", "/health")
