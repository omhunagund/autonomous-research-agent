"""Application-facing workflow execution service."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Lock

from src.core.persistence import ExecutionEvent, PersistenceError, SQLitePersistence
from src.models.schemas import FinalReport, MemoryContext, ResearchState, StoredReport
from src.workflow import build_workflow

logger = logging.getLogger(__name__)


class ExecutionConflictError(RuntimeError):
    """Raised when an execution has already started or finished."""

    def __init__(self, report_id: str, status: str) -> None:
        super().__init__(f"Research execution {report_id} is already in state {status!r}.")
        self.report_id = report_id
        self.status = status


class ExecutionNotFoundError(RuntimeError):
    """Raised when an execution id does not exist."""

    def __init__(self, report_id: str) -> None:
        super().__init__(f"Research execution {report_id} was not found.")
        self.report_id = report_id


class ResearchExecutionError(RuntimeError):
    """A persisted research execution failed after an execution id existed."""

    def __init__(self, report_id: str, cause: Exception) -> None:
        super().__init__(f"Research execution {report_id} failed: {cause}")
        self.report_id = report_id
        self.cause = cause


@dataclass(frozen=True)
class ResearchExecution:
    report_id: str
    report: StoredReport
    memory_context: MemoryContext
    trace: list[ExecutionEvent]


class ResearchExecutionService:
    """Coordinates persistence, automatic memory lookup, workflow execution, and saving."""

    def __init__(self, persistence: SQLitePersistence | None = None) -> None:
        self.persistence = persistence or SQLitePersistence()
        self._execution_lock_guard = Lock()
        self._active_execution_locks: set[str] = set()

    def close(self) -> None:
        """Release service-owned resources. SQLite connections are per-operation."""
        return None

    def _record_trace_best_effort(
        self,
        report_id: str,
        *,
        stage: str,
        event_type: str,
        message: str,
        attempt: int,
    ) -> ExecutionEvent | None:
        try:
            return self.persistence.record_event(
                report_id,
                stage=stage,
                event_type=event_type,
                message=message,
                attempt=attempt,
            )
        except Exception:
            logger.exception(
                "Trace persistence failed for report_id=%s event_type=%s",
                report_id,
                event_type,
            )
            return None

    def _mark_failed_best_effort(self, report_id: str, attempt: int) -> None:
        try:
            self.persistence.mark_failed(report_id)
        except Exception:
            logger.exception("Unable to persist failed status for report_id=%s", report_id)
        self._record_trace_best_effort(
            report_id,
            stage="orchestrator",
            event_type="execution_failed",
            message="Research execution failed before producing a completed report.",
            attempt=attempt,
        )

    def _claim_execution(self, report_id: str) -> None:
        with self._execution_lock_guard:
            if report_id in self._active_execution_locks:
                raise ExecutionConflictError(report_id, "running")
            self._active_execution_locks.add(report_id)

    def _release_execution(self, report_id: str) -> None:
        with self._execution_lock_guard:
            self._active_execution_locks.discard(report_id)

    def initialize(self, topic: str) -> str:
        """Create and persist a new running execution before any workflow work."""
        report_id = self.persistence.create_report(topic)
        # Q194/Q195: the initial trace event is best-effort after the running
        # record has been committed. A trace persistence failure must not abort
        # an otherwise valid execution.
        self._record_trace_best_effort(
            report_id,
            stage="research",
            event_type="research_started",
            message="Research Agent started.",
            attempt=1,
        )
        return report_id

    def execute(self, report_id: str) -> ResearchExecution:
        """Execute exactly one previously initialized running execution."""
        self._claim_execution(report_id)
        try:
            execution_status = self.persistence.get_execution_status(report_id)
            if execution_status is None:
                raise ExecutionNotFoundError(report_id)
            if execution_status != "running":
                raise ExecutionConflictError(report_id, execution_status)

            # The persisted running record is the source of truth for the topic.
            topic = self.persistence.get_execution_topic(report_id)
            if topic is None:
                raise ExecutionNotFoundError(report_id)
            memory = self.persistence.find_memory(topic)

            memory_message = (
                f"Found {len(memory.matches)} relevant prior report(s) for context."
                if memory.matches
                else "No relevant prior reports found."
            )
            self._record_trace_best_effort(
                report_id,
                stage="orchestrator",
                event_type="memory_matches_found",
                message=memory_message,
                attempt=1,
            )

            state = ResearchState(
                report_id=report_id,
                user_topic=topic.strip(),
                memory_context=memory,
                sub_questions=[],
                sources=[],
                findings=[],
                conflicts=[],
                gaps=[],
                research_limitations=[],
                draft=None,
                critique=None,
                retry_count=0,
                revision_target=None,
                final_report=None,
            )

            graph = build_workflow(trace_recorder=self._record_trace_best_effort)
            result = graph.invoke(state.model_dump())
            final_state = ResearchState.model_validate(result)

            if final_state.final_report is None:
                raise RuntimeError("Workflow completed without a final report")

            stored = self.persistence.save_report(report_id, final_state.final_report)
            self._record_trace_best_effort(
                report_id,
                stage="orchestrator",
                event_type="execution_completed",
                message=f"Research execution completed successfully with {stored.quality.value} quality.",
                attempt=final_state.retry_count + 1,
            )
            return ResearchExecution(
                report_id=report_id,
                report=stored,
                memory_context=memory,
                trace=self.persistence.get_trace(report_id),
            )
        except (ExecutionConflictError, ExecutionNotFoundError):
            raise
        except Exception as exc:
            self._mark_failed_best_effort(report_id, max(1, state.retry_count + 1) if "state" in locals() else 1)
            raise ResearchExecutionError(report_id, exc) from exc
        finally:
            self._release_execution(report_id)
