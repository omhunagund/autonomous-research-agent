"""LangGraph workflow definition for the autonomous research agent."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.analysis_agent import analyze_research
from src.agents.critic_agent import critique_report
from src.agents.orchestrator import (
    MAX_CORRECTION_CYCLES,
    _finalize_analysis_only,
    _finalize_unresolved,
    _issue_categories,
    _set_final_quality,
    run_research_correction,
)
from src.agents.research_agent import run_research
from src.agents.writing_agent import write_report
from src.core.llm import get_llm_service
from src.models.schemas import CritiqueCheck, QualityLevel, ResearchState



def _record_trace(
    recorder,
    state: ResearchState,
    *,
    stage: str,
    event_type: str,
    message: str,
    attempt: int | None = None,
) -> None:
    if recorder is None or state.report_id is None:
        return
    recorder(
        state.report_id,
        stage=stage,
        event_type=event_type,
        message=message,
        attempt=state.retry_count + 1 if attempt is None else attempt,
    )


def _research_node(state: ResearchState, recorder=None) -> dict:
    _record_trace(
        recorder, state, stage="research", event_type="research_started",
        message="Research agent is generating sub-questions and gathering web evidence.",
    )
    selection_recovery_subquestions: list[str] = []

    updated = run_research(
        state,
        get_llm_service(),
        on_selection_recovery=selection_recovery_subquestions.append,
    )

    for sub_question in selection_recovery_subquestions:
        _record_trace(
            recorder,
            updated,
            stage="research",
            event_type="search_selection_recovered",
            message=(
                "Search-candidate selection for "
                f'"{sub_question}" required local recovery from malformed '
                "structured output; recovered indices passed deterministic validation."
            ),
        )

    _record_trace(
        recorder, updated, stage="research", event_type="research_completed",
        message=(
            f"Research produced {len(updated.sub_questions)} sub-question(s) and "
            f"{len(updated.sources)} unique usable source(s)."
        ),
    )
    analysis = analyze_research(updated, get_llm_service())
    _record_trace(
        recorder, updated, stage="analysis", event_type="analysis_completed",
        message=(
            f"Analysis evaluated {len(analysis.findings)} finding(s), "
            f"{len(analysis.gaps)} gap(s), and {len(analysis.conflicts)} conflict(s)."
        ),
    )
    return {
        "sub_questions": updated.sub_questions,
        "sources": updated.sources,
        "research_limitations": updated.research_limitations,
        "findings": analysis.findings,
        "conflicts": analysis.conflicts,
        "gaps": analysis.gaps,
    }


def _writing_node(state: ResearchState, recorder=None) -> dict:
    _record_trace(
        recorder, state, stage="writing", event_type="writing_started",
        message="Writing agent is composing the cited research report.",
    )
    report = write_report(state, get_llm_service())
    _record_trace(
        recorder, state, stage="writing", event_type="writing_completed",
        message="Writing agent completed the current report draft.",
    )
    return {"draft": report}


def _critic_node(state: ResearchState, recorder=None) -> dict:
    _record_trace(
        recorder, state, stage="critic", event_type="critic_review_started",
        message="Critic is reviewing faithfulness, coverage, recency, and balance.",
    )
    critique, target = critique_report(state, get_llm_service())
    _record_trace(
        recorder, state, stage="critic", event_type="critic_review_completed",
        message=(
            f"Critic verdict: {critique.verdict.value.upper()}; "
            f"correction target: {target}."
        ),
    )
    return {"critique": critique, "revision_target": target}


def _route_after_critic(state: ResearchState) -> str:
    critique = state.critique
    if critique is None:
        raise RuntimeError("Critic route requires a Critique in state")

    if critique.verdict == CritiqueCheck.PASS:
        return "pass"

    categories = _issue_categories(critique.issues)
    if categories and categories <= {"ANALYSIS", "INFO"} and "ANALYSIS" in categories:
        return "analysis_finalize"

    if state.revision_target in {"research", "writing"} and state.retry_count < MAX_CORRECTION_CYCLES:
        return state.revision_target

    return "unresolved_finalize"


def _research_correction_node(state: ResearchState, recorder=None) -> dict:
    if state.critique is None:
        raise RuntimeError("Research correction requires the latest Critique")

    _record_trace(
        recorder, state, stage="research", event_type="research_correction_started",
        message=(
            f"Starting targeted Research correction cycle {state.retry_count + 1} "
            "from the latest Critic issues."
        ),
        attempt=state.retry_count + 1,
    )
    research_issues = [
        issue for issue in state.critique.issues if issue.startswith("[RESEARCH]")
    ]
    updated, recovered_query_generation, recovered_selection = run_research_correction(
        state.model_copy(update={"retry_count": state.retry_count + 1}),
        research_issues,
        get_llm_service(),
    )
    if recovered_query_generation:
        _record_trace(
            recorder,
            state,
            stage="research",
            event_type="correction_query_generation_recovered",
            message=(
                "Correction-query generation required local recovery from "
                "malformed structured output; recovered queries passed validation."
            ),
            attempt=state.retry_count + 1,
        )
    if recovered_selection:
        _record_trace(
            recorder,
            updated,
            stage="research",
            event_type="search_selection_recovered",
            message=(
                "Search-candidate selection during Research correction required "
                "local recovery from malformed structured output; recovered indices "
                "passed deterministic validation."
            ),
            attempt=state.retry_count + 1,
        )
    analysis = analyze_research(updated, get_llm_service())
    refreshed = updated.model_copy(
        update={
            "findings": analysis.findings,
            "conflicts": analysis.conflicts,
            "gaps": analysis.gaps,
            "draft": None,
            "critique": None,
            "revision_target": None,
        }
    )
    _record_trace(
        recorder, refreshed, stage="research",
        event_type="research_correction_completed",
        message=(
            f"Research correction cycle {refreshed.retry_count} completed; "
            f"current evidence contains {len(refreshed.sources)} usable source(s)."
        ),
        attempt=refreshed.retry_count,
    )
    return {
        "retry_count": refreshed.retry_count,
        "sources": refreshed.sources,
        "research_limitations": refreshed.research_limitations,
        "findings": refreshed.findings,
        "conflicts": refreshed.conflicts,
        "gaps": refreshed.gaps,
    }


def _writing_correction_node(state: ResearchState, recorder=None) -> dict:
    if state.critique is None:
        raise RuntimeError("Writing correction requires the latest Critique")

    _record_trace(
        recorder, state, stage="writing", event_type="writing_correction_started",
        message=(
            f"Starting targeted Writing correction cycle {state.retry_count + 1} "
            "from the latest Critic issues."
        ),
        attempt=state.retry_count + 1,
    )
    writing_issues = [
        issue for issue in state.critique.issues if issue.startswith("[WRITING]")
    ]
    report = write_report(
        state.model_copy(update={"retry_count": state.retry_count + 1}),
        get_llm_service(),
        correction_issues=writing_issues,
    )
    _record_trace(
        recorder, state, stage="writing",
        event_type="writing_correction_completed",
        message="Writing correction completed and produced a revised draft.",
        attempt=state.retry_count + 1,
    )
    return {"retry_count": state.retry_count + 1, "draft": report}


def _finalize_pass(state: ResearchState, recorder=None) -> dict:
    quality = QualityLevel.HIGH if state.retry_count == 0 else QualityLevel.MEDIUM
    finalized = _set_final_quality(state, quality)
    return {"draft": finalized.draft, "final_report": finalized.final_report, "revision_target": "none"}


def _finalize_analysis(state: ResearchState, recorder=None) -> dict:
    finalized = _finalize_analysis_only(state, state.critique)  # type: ignore[arg-type]
    return {
        "draft": finalized.draft,
        "final_report": finalized.final_report,
        "revision_target": finalized.revision_target,
    }


def _finalize_unresolved_node(state: ResearchState, recorder=None) -> dict:
    finalized = _finalize_unresolved(state, state.critique)  # type: ignore[arg-type]
    return {
        "draft": finalized.draft,
        "final_report": finalized.final_report,
        "revision_target": finalized.revision_target,
    }


def build_workflow(trace_recorder=None):
    """Build the bounded Research -> Analysis -> Writing -> Critic graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("research", lambda state: _research_node(state, trace_recorder))
    graph.add_node("writing", lambda state: _writing_node(state, trace_recorder))
    graph.add_node("critic", lambda state: _critic_node(state, trace_recorder))
    graph.add_node(
        "research_correction",
        lambda state: _research_correction_node(state, trace_recorder),
    )
    graph.add_node(
        "writing_correction",
        lambda state: _writing_correction_node(state, trace_recorder),
    )
    graph.add_node("finalize_pass", lambda state: _finalize_pass(state, trace_recorder))
    graph.add_node(
        "finalize_analysis",
        lambda state: _finalize_analysis(state, trace_recorder),
    )
    graph.add_node(
        "finalize_unresolved",
        lambda state: _finalize_unresolved_node(state, trace_recorder),
    )

    graph.add_edge(START, "research")
    graph.add_edge("research", "writing")
    graph.add_edge("writing", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "pass": "finalize_pass",
            "research": "research_correction",
            "writing": "writing_correction",
            "analysis_finalize": "finalize_analysis",
            "unresolved_finalize": "finalize_unresolved",
        },
    )
    graph.add_edge("research_correction", "writing")
    graph.add_edge("writing_correction", "critic")
    graph.add_edge("finalize_pass", END)
    graph.add_edge("finalize_analysis", END)
    graph.add_edge("finalize_unresolved", END)

    return graph.compile()
