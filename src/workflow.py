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


def _research_node(state: ResearchState) -> dict:
    updated = run_research(state, get_llm_service())
    analysis = analyze_research(updated, get_llm_service())
    return {
        "sub_questions": updated.sub_questions,
        "sources": updated.sources,
        "research_limitations": updated.research_limitations,
        "findings": analysis.findings,
        "conflicts": analysis.conflicts,
        "gaps": analysis.gaps,
    }


def _writing_node(state: ResearchState) -> dict:
    return {"draft": write_report(state, get_llm_service())}


def _critic_node(state: ResearchState) -> dict:
    critique, target = critique_report(state, get_llm_service())
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


def _research_correction_node(state: ResearchState) -> dict:
    if state.critique is None:
        raise RuntimeError("Research correction requires the latest Critique")

    research_issues = [
        issue for issue in state.critique.issues if issue.startswith("[RESEARCH]")
    ]
    updated = run_research_correction(
        state.model_copy(update={"retry_count": state.retry_count + 1}),
        research_issues,
        get_llm_service(),
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
    return {
        "retry_count": refreshed.retry_count,
        "sources": refreshed.sources,
        "research_limitations": refreshed.research_limitations,
        "findings": refreshed.findings,
        "conflicts": refreshed.conflicts,
        "gaps": refreshed.gaps,
    }


def _writing_correction_node(state: ResearchState) -> dict:
    if state.critique is None:
        raise RuntimeError("Writing correction requires the latest Critique")

    writing_issues = [
        issue for issue in state.critique.issues if issue.startswith("[WRITING]")
    ]
    report = write_report(
        state.model_copy(update={"retry_count": state.retry_count + 1}),
        get_llm_service(),
        correction_issues=writing_issues,
    )
    return {"retry_count": state.retry_count + 1, "draft": report}


def _finalize_pass(state: ResearchState) -> dict:
    quality = QualityLevel.HIGH if state.retry_count == 0 else QualityLevel.MEDIUM
    finalized = _set_final_quality(state, quality)
    return {"draft": finalized.draft, "final_report": finalized.final_report, "revision_target": "none"}


def _finalize_analysis(state: ResearchState) -> dict:
    finalized = _finalize_analysis_only(state, state.critique)  # type: ignore[arg-type]
    return {
        "draft": finalized.draft,
        "final_report": finalized.final_report,
        "revision_target": finalized.revision_target,
    }


def _finalize_unresolved_node(state: ResearchState) -> dict:
    finalized = _finalize_unresolved(state, state.critique)  # type: ignore[arg-type]
    return {
        "draft": finalized.draft,
        "final_report": finalized.final_report,
        "revision_target": finalized.revision_target,
    }


def build_workflow():
    """Build the bounded Research -> Analysis -> Writing -> Critic graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("research", _research_node)
    graph.add_node("writing", _writing_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("research_correction", _research_correction_node)
    graph.add_node("writing_correction", _writing_correction_node)
    graph.add_node("finalize_pass", _finalize_pass)
    graph.add_node("finalize_analysis", _finalize_analysis)
    graph.add_node("finalize_unresolved", _finalize_unresolved_node)

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
