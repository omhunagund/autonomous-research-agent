"""Application-facing workflow execution service."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.persistence import ExecutionEvent, SQLitePersistence
from src.models.schemas import MemoryContext, ResearchState, StoredReport
from src.workflow import build_workflow


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

    def run(self, topic: str) -> ResearchExecution:
        report_id = self.persistence.create_report(topic)
        self.persistence.record_event(
            report_id,
            stage="orchestrator",
            event_type="execution_started",
            message="Starting autonomous research workflow.",
        )

        memory = self.persistence.find_memory(topic)
        memory_message = (
            f"Found {len(memory.matches)} relevant prior report(s) for context."
            if memory.matches
            else "No relevant prior reports found."
        )
        self.persistence.record_event(
            report_id,
            stage="memory",
            event_type="memory_matches_found",
            message=memory_message,
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

        graph = build_workflow()
        result = graph.invoke(state.model_dump())
        final_state = ResearchState.model_validate(result)
        if final_state.final_report is None:
            raise RuntimeError("Workflow completed without a final report")

        stored = self.persistence.save_report(report_id, final_state.final_report)
        self.persistence.record_event(
            report_id,
            stage="orchestrator",
            event_type="execution_completed",
            message=f"Final report completed with {stored.quality.value} quality.",
            attempt=final_state.retry_count,
        )
        return ResearchExecution(
            report_id=report_id,
            report=stored,
            memory_context=memory,
            trace=self.persistence.get_trace(report_id),
        )
