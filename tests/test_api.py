from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.execution import ResearchExecution, ResearchExecutionError
from src.core.persistence import ExecutionEvent, SQLitePersistence
from src.models.schemas import FinalReport, StoredReport, QualityLevel


def _report(topic: str = "Test topic") -> FinalReport:
    return FinalReport(
        topic=topic,
        executive_summary="A concise summary.",
        key_findings=[],
        supporting_evidence=[],
        gaps=[],
        gap_explanations=[],
        conflicts=[],
        conflict_explanations=[],
        quality=QualityLevel.HIGH,
        references=[],
        analysis_issues=[],
        unresolved_issues=[],
    )


class FakeService:
    def __init__(self, tmp_path: Path) -> None:
        self.persistence = SQLitePersistence(tmp_path / "research.db")
        self.report_id = self.persistence.create_report("Test topic")
        stored = self.persistence.save_report(self.report_id, _report("Test topic"))
        self.stored = stored

        self.persistence.record_event(
            self.report_id,
            stage="research",
            event_type="research_started",
            message="Research Agent started.",
            attempt=1,
        )

        self.persistence.record_event(
            self.report_id,
            stage="orchestrator",
            event_type="execution_completed",
            message="Research execution completed successfully with high quality.",
            attempt=1,
        )

    def run(self, topic: str) -> ResearchExecution:
        return ResearchExecution(
            report_id=self.report_id,
            report=self.stored,
            memory_context=self.persistence.find_memory(topic),
            trace=self.persistence.get_trace(self.report_id),
        )

    def close(self) -> None:
        return None


def _client(service) -> TestClient:
    app.dependency_overrides.clear()
    from src.api.routes import get_execution_service
    app.dependency_overrides[get_execution_service] = lambda: service
    return TestClient(app)


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_health_returns_ok(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_response_shape(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        response = client.post("/research", json={"topic": "another topic"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"report_id", "report"}
    assert body["report"]["quality"] == "high"


def test_blank_topic_returns_structured_422(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        response = client.post("/research", json={"topic": "   "})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ValidationError"
    assert body["report_id"] is None


def test_get_report_and_history(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        report_response = client.get(f"/reports/{service.report_id}")
        history_response = client.get("/reports/history")
    assert report_response.status_code == 200
    assert report_response.json()["report_id"] == service.report_id
    history = history_response.json()["reports"]
    assert len(history) == 1
    assert set(history[0]) == {"report_id", "topic", "created_at", "quality"}


def test_get_report_unknown_and_malformed_ids(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        missing = client.get("/reports/550e8400-e29b-41d4-a716-446655440000")
        malformed = client.get("/reports/not-a-uuid")
    assert missing.status_code == 404
    assert missing.json()["error"] == "ReportNotFound"
    assert malformed.status_code == 422
    assert malformed.json()["error"] == "ValidationError"


def test_trace_returns_status_and_events(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        response = client.get(f"/research/{service.report_id}/trace")
    assert response.status_code == 200
    body = response.json()
    assert body["report_id"] == service.report_id
    assert body["status"] == "completed"
    assert body["events"][0]["stage"] == "research"
    assert body["events"][0]["event_type"] == "research_started"
    assert body["events"][0]["attempt"] == 1
    assert body["events"][-1]["event_type"] == "execution_completed"
    assert body["events"][-1]["attempt"] == 1


def test_trace_unknown_id_returns_404(tmp_path: Path) -> None:
    service = FakeService(tmp_path)
    with _client(service) as client:
        response = client.get("/research/550e8400-e29b-41d4-a716-446655440000/trace")
    assert response.status_code == 404
    assert response.json()["error"] == "ReportNotFound"


def test_research_execution_upstream_failure_returns_502_with_report_id(tmp_path: Path) -> None:
    class FailingService(FakeService):
        def run(self, topic: str) -> ResearchExecution:
            from src.core.llm import LLMInvocationError
            raise ResearchExecutionError(self.report_id, LLMInvocationError("provider failed"))

    service = FailingService(tmp_path)
    with _client(service) as client:
        response = client.post("/research", json={"topic": "test"})
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "UpstreamDependencyError"
    assert body["report_id"] == service.report_id


def test_research_execution_unexpected_failure_returns_500_with_report_id(tmp_path: Path) -> None:
    class FailingService(FakeService):
        def run(self, topic: str) -> ResearchExecution:
            raise ResearchExecutionError(self.report_id, RuntimeError("unexpected"))

    service = FailingService(tmp_path)
    with _client(service) as client:
        response = client.post("/research", json={"topic": "test"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalServerError"
    assert body["report_id"] == service.report_id


def test_history_is_empty_without_completed_reports(tmp_path: Path) -> None:
    class EmptyService:
        def __init__(self) -> None:
            self.persistence = SQLitePersistence(tmp_path / "empty.db")

    service = EmptyService()
    with _client(service) as client:
        response = client.get("/reports/history")
    assert response.status_code == 200
    assert response.json() == {"reports": []}
