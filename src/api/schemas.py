"""Public HTTP request/response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, field_validator

from src.models.schemas import FinalReport, QualityLevel


class ResearchRequest(BaseModel):
    topic: str

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Research topic cannot be blank")
        return value


class ResearchResponse(BaseModel):
    report_id: str
    report: FinalReport


class HistoryItem(BaseModel):
    report_id: str
    topic: str
    created_at: datetime
    quality: QualityLevel


class HistoryResponse(BaseModel):
    reports: list[HistoryItem]


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceEventType(str, Enum):
    EXECUTION_STARTED = "execution_started"
    MEMORY_MATCHES_FOUND = "memory_matches_found"
    RESEARCH_STARTED = "research_started"
    RESEARCH_COMPLETED = "research_completed"
    RESEARCH_CORRECTION_STARTED = "research_correction_started"
    RESEARCH_CORRECTION_COMPLETED = "research_correction_completed"
    ANALYSIS_COMPLETED = "analysis_completed"
    WRITING_STARTED = "writing_started"
    WRITING_COMPLETED = "writing_completed"
    WRITING_CORRECTION_STARTED = "writing_correction_started"
    WRITING_CORRECTION_COMPLETED = "writing_correction_completed"
    CRITIC_REVIEW_STARTED = "critic_review_started"
    CRITIC_REVIEW_COMPLETED = "critic_review_completed"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"


class TraceEvent(BaseModel):
    event_id: str
    timestamp: datetime
    stage: str
    event_type: TraceEventType
    message: str
    attempt: int


class TraceResponse(BaseModel):
    report_id: str
    status: ExecutionStatus
    events: list[TraceEvent]


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None
    report_id: str | None = None
