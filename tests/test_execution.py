from __future__ import annotations

from pathlib import Path
from threading import Barrier, Thread

import pytest

from src.core.execution import (
    ExecutionConflictError,
    ExecutionNotFoundError,
    ResearchExecutionError,
    ResearchExecutionService,
)
from src.core.persistence import SQLitePersistence
from src.models.schemas import (
    FinalReport,
    QualityLevel,
    ResearchState,
)


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


class FakeGraph:
    def __init__(self, result_factory=None, error: Exception | None = None) -> None:
        self.result_factory = result_factory
        self.error = error
        self.invocations = 0

    def invoke(self, payload):
        self.invocations += 1

        if self.error is not None:
            raise self.error

        if self.result_factory is not None:
            return self.result_factory(payload)

        state = ResearchState.model_validate(payload)
        return state.model_copy(
            update={
                "final_report": _report(state.user_topic),
                "draft": _report(state.user_topic),
            }
        ).model_dump()


def test_initialize_creates_running_execution(tmp_path: Path) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("AI in healthcare")

    assert persistence.get_execution_status(report_id) == "running"
    assert persistence.get_execution_topic(report_id) == "AI in healthcare"

    trace = persistence.get_trace(report_id)

    assert len(trace) == 1
    assert trace[0].report_id == report_id
    assert trace[0].stage == "research"
    assert trace[0].event_type == "research_started"
    assert trace[0].attempt == 1


def test_execute_completes_and_persists_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("AI in healthcare")

    graph = FakeGraph()

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: graph,
    )

    execution = service.execute(report_id)

    assert graph.invocations == 1
    assert execution.report_id == report_id
    assert execution.report.quality is QualityLevel.HIGH

    assert persistence.get_execution_status(report_id) == "completed"

    stored = persistence.get_report(report_id)

    assert stored is not None
    assert stored.report.topic == "AI in healthcare"
    assert stored.quality is QualityLevel.HIGH

    trace = persistence.get_trace(report_id)

    assert trace[-1].event_type == "execution_completed"


def test_execute_rejects_completed_execution(
    tmp_path: Path,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = persistence.create_report("Already completed")
    persistence.save_report(report_id, _report("Already completed"))

    with pytest.raises(ExecutionConflictError) as exc_info:
        service.execute(report_id)

    assert exc_info.value.report_id == report_id
    assert exc_info.value.status == "completed"


def test_execute_rejects_unknown_execution(
    tmp_path: Path,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = "550e8400-e29b-41d4-a716-446655440000"

    with pytest.raises(ExecutionNotFoundError) as exc_info:
        service.execute(report_id)

    assert exc_info.value.report_id == report_id


def test_workflow_failure_marks_execution_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("Failure topic")

    graph = FakeGraph(error=RuntimeError("workflow failed"))

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: graph,
    )

    with pytest.raises(ResearchExecutionError) as exc_info:
        service.execute(report_id)

    assert exc_info.value.report_id == report_id
    assert isinstance(exc_info.value.cause, RuntimeError)

    assert persistence.get_execution_status(report_id) == "failed"

    trace = persistence.get_trace(report_id)

    assert trace[-1].event_type == "execution_failed"
    assert trace[-1].stage == "orchestrator"


def test_failed_execution_is_not_available_as_completed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("Failed topic")

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: FakeGraph(
            error=RuntimeError("workflow failed")
        ),
    )

    with pytest.raises(ResearchExecutionError):
        service.execute(report_id)

    assert persistence.get_report(report_id) is None
    assert persistence.list_reports() == []
    assert persistence.find_memory("Failed topic").matches == []


def test_trace_persistence_failure_does_not_abort_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("Trace failure topic")

    original_record_event = persistence.record_event

    call_count = {"value": 0}

    def flaky_record_event(*args, **kwargs):
        call_count["value"] += 1

        # Allow the first event created during initialize().
        if call_count["value"] >= 2:
            raise RuntimeError("trace database unavailable")

        return original_record_event(*args, **kwargs)

    monkeypatch.setattr(
        persistence,
        "record_event",
        flaky_record_event,
    )

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: FakeGraph(),
    )

    execution = service.execute(report_id)

    assert execution.report_id == report_id
    assert persistence.get_execution_status(report_id) == "completed"

    stored = persistence.get_report(report_id)

    assert stored is not None
    assert stored.quality is QualityLevel.HIGH

def test_same_report_id_concurrent_execute_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    report_id = service.initialize("Concurrent same report")

    entered = Barrier(2)
    release = Barrier(2)
    execution_errors: list[Exception] = []

    class BlockingGraph:
        def invoke(self, payload):
            entered.wait(timeout=5)
            release.wait(timeout=5)

            state = ResearchState.model_validate(payload)
            report = _report(state.user_topic)

            return state.model_copy(
                update={
                    "final_report": report,
                    "draft": report,
                }
            ).model_dump()

    graph = BlockingGraph()

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: graph,
    )

    first_result: list[ResearchExecution] = []

    def run_first() -> None:
        try:
            first_result.append(service.execute(report_id))
        except Exception as exc:
            execution_errors.append(exc)

    thread = Thread(target=run_first)
    thread.start()

    entered.wait(timeout=5)

    with pytest.raises(ExecutionConflictError) as exc_info:
        service.execute(report_id)

    assert exc_info.value.report_id == report_id
    assert exc_info.value.status == "running"

    release.wait(timeout=5)
    thread.join(timeout=5)

    assert not execution_errors
    assert len(first_result) == 1
    assert persistence.get_execution_status(report_id) == "completed"

def test_different_report_ids_can_execute_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = SQLitePersistence(tmp_path / "research.db")
    service = ResearchExecutionService(persistence)

    first_report_id = service.initialize("Concurrent topic one")
    second_report_id = service.initialize("Concurrent topic two")

    entered = Barrier(3)
    release = Barrier(3)

    completed_ids: list[str] = []
    execution_errors: list[Exception] = []

    class BlockingGraph:
        def invoke(self, payload):
            # Main thread + both execution threads must reach this point.
            entered.wait(timeout=5)

            # Do not let either workflow finish until both executions
            # have reached the workflow.
            release.wait(timeout=5)

            state = ResearchState.model_validate(payload)
            report = _report(state.user_topic)

            return state.model_copy(
                update={
                    "final_report": report,
                    "draft": report,
                }
            ).model_dump()

    monkeypatch.setattr(
        "src.core.execution.build_workflow",
        lambda trace_recorder: BlockingGraph(),
    )

    def run(report_id: str) -> None:
        try:
            service.execute(report_id)
            completed_ids.append(report_id)
        except Exception as exc:
            execution_errors.append(exc)

    first_thread = Thread(target=run, args=(first_report_id,))
    second_thread = Thread(target=run, args=(second_report_id,))

    first_thread.start()
    second_thread.start()

    # Main thread joins the first barrier. This cannot pass until BOTH
    # worker threads have reached BlockingGraph.invoke().
    entered.wait(timeout=5)

    # Release both workers simultaneously.
    release.wait(timeout=5)

    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not execution_errors
    assert set(completed_ids) == {first_report_id, second_report_id}

    assert persistence.get_execution_status(first_report_id) == "completed"
    assert persistence.get_execution_status(second_report_id) == "completed"