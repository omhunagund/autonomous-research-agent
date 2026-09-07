"""Streamlit live research workspace."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import re
from typing import Any

import streamlit as st

from frontend.api_client import ResearchAPIClient
from frontend.models import APIClientError, ActiveExecution, ExecutionStatus, QualityLevel, TraceEvent, TraceResponse


POLL_INTERVAL_SECONDS = 1
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="research-exec")

LANDING_COPY = (
    "Autonomous Research & Report Agent — Enter a research topic and let a "
    "multi-agent workflow research the web, analyze evidence, write a cited "
    "report, and critique it for gaps, conflicts, and quality."
)

ATTEMPT_LABELS = {
    1: "Attempt 1 — Initial research",
    2: "Attempt 2 — Research correction",
    3: "Attempt 3 — Writing correction",
}

STAGE_LABELS = {
    "orchestrator": "Orchestrator",
    "research": "Research",
    "analysis": "Analysis",
    "writing": "Writing",
    "critic": "Critic",
}

STAGE_EVENT_TYPES = {
    "research": {"research_started", "research_completed", "research_correction_started", "research_correction_completed"},
    "analysis": {"analysis_completed"},
    "writing": {"writing_started", "writing_completed", "writing_correction_started", "writing_correction_completed"},
    "critic": {"critic_review_started", "critic_review_completed"},
}


def _configure_page() -> None:
    st.set_page_config(
        page_title="Autonomous Research & Report Agent",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        [data-testid="stSidebar"] {
            background: #f7f8fa;
            border-right: 1px solid #e5e7eb;
        }
        .research-shell {
            max-width: 860px;
            margin: 4rem auto 2rem auto;
        }
        .research-title {
            font-size: 2.25rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.85rem;
            color: #111827;
        }
        .research-copy {
            color: #6b7280;
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 1.75rem;
        }
        .stTextInput input:focus {
            border-color: #9ca3af !important;
            box-shadow: 0 0 0 1px rgba(156, 163, 175, 0.22) !important;
        }
        .live-header {
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }
        .attempt-heading {
            font-weight: 700;
            color: #111827;
            margin: 1rem 0 0.75rem;
        }
        .stepper {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
        }
        .step {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            min-height: 30px;
            padding: 0.25rem 0.55rem;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #4b5563;
            background: #ffffff;
            font-size: 0.84rem;
        }
        .step.current {
            border-color: #9ca3af;
            box-shadow: 0 1px 4px rgba(17, 24, 39, 0.08);
            color: #111827;
        }
        .step.complete {
            color: #111827;
            background: #f3f4f6;
        }
        .connector {
            color: #9ca3af;
        }
        .trace-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 0.1rem 0 0.65rem;
        }
        .jump-latest {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.65rem;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #374151 !important;
            background: #ffffff;
            text-decoration: none !important;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .jump-latest:hover {
            border-color: #9ca3af;
            color: #111827 !important;
            background: #f9fafb;
        }
        .trace-new-events {
            color: #6b7280;
            font-size: 0.82rem;
        }
        .trace-row {
            display: flex;
            gap: 0.75rem;
            padding: 0.55rem 0;
            border-bottom: 1px solid #f0f1f3;
        }
        .trace-stage {
            min-width: 88px;
            color: #374151;
            font-weight: 600;
        }
        .trace-message {
            color: #4b5563;
        }
        .report-shell {
            max-width: 900px;
        }
        .report-section {
            margin: 2rem 0 1.6rem;
        }
        .report-summary {
            font-size: 1.02rem;
            line-height: 1.75;
            color: #1f2937;
        }
        .report-item {
            padding: 0.95rem 1rem;
            margin: 0.8rem 0;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #ffffff;
        }
        .report-item-title {
            color: #111827;
            font-weight: 650;
            line-height: 1.55;
        }
        .confidence-badge, .type-badge {
            display: inline-block;
            margin-left: 0.5rem;
            padding: 0.12rem 0.42rem;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #4b5563;
            background: #ffffff;
            font-size: 0.74rem;
            font-weight: 600;
            vertical-align: middle;
        }
        .report-body {
            margin-top: 0.45rem;
            color: #374151;
            line-height: 1.7;
        }
        .report-meta {
            margin-top: 0.55rem;
            color: #6b7280;
            font-size: 0.84rem;
            line-height: 1.55;
        }
        .report-link {
            color: #374151;
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .report-link:hover {
            color: #111827;
        }
        .source-card {
            scroll-margin-top: 1.25rem;
            padding: 0.9rem 1rem;
            margin: 0.7rem 0;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #fafafa;
        }
        .source-card:target {
            border-color: #9ca3af;
            box-shadow: 0 0 0 2px rgba(156, 163, 175, 0.15);
            background: #f9fafb;
        }
        .source-number {
            display: inline-block;
            min-width: 1.65rem;
            font-weight: 700;
            color: #111827;
        }
        .relation-link {
            display: inline-block;
            margin-top: 0.55rem;
            margin-right: 0.45rem;
            padding: 0.15rem 0.42rem;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            color: #4b5563;
            text-decoration: none;
            font-size: 0.79rem;
        }
        .relation-link:hover {
            border-color: #9ca3af;
            color: #111827;
        }
        .report-empty {
            color: #6b7280;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _quality_label(value: QualityLevel) -> str:
    return value.value.capitalize()

def _history_topic_label(topic: str, max_length: int = 42) -> str:
    """Return a compact history label while preserving the full topic in the tooltip."""
    topic = topic.strip()

    if len(topic) <= max_length:
        return topic

    return f"{topic[:max_length - 1].rstrip()}…"


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "workspace_mode": "landing",
        "history_loaded": False,
        "history": [],
        "history_error": None,
        "active_research": [],
        "active_error": None,
        "selected_report_id": None,
        "selected_topic": None,
        "selected_quality": None,
        "current_trace": None,
        "current_report": None,
        "execution_error": None,
        "report_error": None,
        "polling_active": False,
        "new_research_requested": False,
        "trace_seen_event_ids": tuple(),
        "trace_has_new_events": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _submit_execution(client: ResearchAPIClient, report_id: str) -> Future:
    return _EXECUTOR.submit(client.execute_research, report_id)


def _start_research(client: ResearchAPIClient, topic: str) -> None:
    try:
        init = client.initialize_research(topic)
    except APIClientError as exc:
        st.session_state["execution_error"] = exc.message
        st.session_state["workspace_mode"] = "landing"
        return

    st.session_state["selected_report_id"] = init.report_id
    st.session_state["selected_topic"] = topic
    st.session_state["selected_quality"] = None
    st.session_state["current_trace"] = TraceResponse(
        report_id=init.report_id,
        status=ExecutionStatus.RUNNING,
        events=[],
    )
    st.session_state["current_report"] = None
    st.session_state["execution_error"] = None
    st.session_state["trace_seen_event_ids"] = tuple()
    st.session_state["trace_has_new_events"] = False
    st.session_state["report_error"] = None
    st.session_state["workspace_mode"] = "live"
    st.session_state["polling_active"] = True
    _submit_execution(client, init.report_id)
    _refresh_active(client, select_id=init.report_id)


def _refresh_history(client: ResearchAPIClient) -> None:
    try:
        st.session_state["history"] = client.get_history()
        st.session_state["history_error"] = None
        st.session_state["history_loaded"] = True
    except APIClientError as exc:
        st.session_state["history_error"] = exc.message
        st.session_state["history_loaded"] = True


def _refresh_active(client: ResearchAPIClient, *, select_id: str | None = None) -> None:
    try:
        active = client.get_active_research()
        st.session_state["active_research"] = active
        st.session_state["active_error"] = None
        if select_id is not None and any(item.report_id == select_id for item in active):
            st.session_state["selected_report_id"] = select_id
            st.session_state["selected_topic"] = next(
                item.topic for item in active if item.report_id == select_id
            )
    except APIClientError as exc:
        st.session_state["active_error"] = exc.message


def _ensure_history(client: ResearchAPIClient) -> None:
    if not st.session_state["history_loaded"]:
        _refresh_history(client)


def _render_sidebar(client: ResearchAPIClient) -> None:
    with st.sidebar:
        st.subheader("Research History")

        active = st.session_state["active_research"]
        if active or st.session_state["active_error"]:
            st.markdown("**Active Research**")
            if st.session_state["active_error"]:
                st.error(st.session_state["active_error"])
                if st.button("Retry", key="active_retry", use_container_width=True):
                    _refresh_active(client)
                    st.rerun()
            for item in active:
                label = f"{item.topic}  \nRunning · Attempt {item.attempt}"
                if st.button(
                    label,
                    key=f"active_{item.report_id}",
                    use_container_width=True,
                ):
                    _select_active(client, item)
                    st.rerun()

        if st.session_state["history_error"]:
            st.error(st.session_state["history_error"])
            if st.button("Retry", key="history_retry", use_container_width=True):
                st.session_state["history_loaded"] = False
                _refresh_history(client)
                st.rerun()

        history = st.session_state["history"]
        if not history and not st.session_state["history_error"]:
            st.caption("No completed research yet.")

        for item in history:
            topic = item.topic
            selected = item.report_id == st.session_state.get("selected_report_id") and st.session_state.get("workspace_mode") == "history"
            label = f"{'• ' if selected else ''}{_history_topic_label(topic)}"
            if st.button(
                label,
                key=f"history_{item.report_id}",
                use_container_width=True,
                help=topic,
            ):
                _select_history(client, item.report_id, topic, item.quality)
                st.rerun()
            st.caption(f"{_quality_label(item.quality)} · {item.created_at.astimezone().strftime('%d %b %Y')}")


def _select_history(
    client: ResearchAPIClient,
    report_id: str,
    topic: str,
    quality: QualityLevel,
) -> None:
    try:
        payload = client.get_report(report_id)
    except APIClientError as exc:
        st.session_state["report_error"] = exc.message
        return

    st.session_state["selected_report_id"] = report_id
    st.session_state["selected_topic"] = topic
    st.session_state["selected_quality"] = quality
    st.session_state["current_report"] = payload["report"]
    st.session_state["current_trace"] = None
    st.session_state["workspace_mode"] = "history"
    st.session_state["report_error"] = None


def _select_active(client: ResearchAPIClient, item: ActiveExecution) -> None:
    st.session_state["selected_report_id"] = item.report_id
    st.session_state["selected_topic"] = item.topic
    st.session_state["selected_quality"] = None
    st.session_state["workspace_mode"] = "live"
    st.session_state["current_report"] = None
    st.session_state["execution_error"] = None
    st.session_state["report_error"] = None
    st.session_state["polling_active"] = True
    st.session_state["current_trace"] = None
    st.session_state["trace_seen_event_ids"] = tuple()
    st.session_state["trace_has_new_events"] = False
    try:
        st.session_state["current_trace"] = client.get_trace(item.report_id)
    except APIClientError as exc:
        st.session_state["execution_error"] = exc.message


def _render_landing(client: ResearchAPIClient) -> None:
    st.markdown('<div class="research-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="research-title">Autonomous Research &amp; Report Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="research-copy">{html.escape(LANDING_COPY)}</div>',
        unsafe_allow_html=True,
    )
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. How is generative AI changing software engineering?",
        label_visibility="collapsed",
        key="topic_input",
    )
    if st.button(
        "Research",
        type="primary",
        disabled=not topic.strip(),
        key="research_button",
    ):
        _start_research(client, topic.strip())
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _event_completed(events: list[TraceEvent], stage: str) -> bool:
    terminal = {
        "research": {"research_completed", "research_correction_completed"},
        "analysis": {"analysis_completed"},
        "writing": {"writing_completed", "writing_correction_completed"},
        "critic": {"critic_review_completed"},
    }.get(stage, set())
    return any(event.event_type in terminal for event in events)


def _current_stage(events: list[TraceEvent]) -> str | None:
    for event in reversed(events):
        if event.stage in {"research", "analysis", "writing", "critic"}:
            return event.stage
    return None


def _render_stepper(events: list[TraceEvent], attempt: int) -> None:
    attempt_events = [event for event in events if event.attempt == attempt]
    current = _current_stage(attempt_events)

    st.markdown(
        f'<div class="attempt-heading">{html.escape(ATTEMPT_LABELS.get(attempt, f"Attempt {attempt}"))}</div>',
        unsafe_allow_html=True,
    )
    stages = ["research", "analysis", "writing", "critic"]
    chunks: list[str] = ['<div class="stepper">']
    for index, stage in enumerate(stages):
        complete = _event_completed(attempt_events, stage)
        cls = "step complete" if complete else ("step current" if current == stage else "step")
        icon = "✓" if complete else ("●" if current == stage else "○")
        chunks.append(
            f'<span class="{cls}">{icon} {html.escape(STAGE_LABELS[stage])}</span>'
        )
        if index < len(stages) - 1:
            chunks.append('<span class="connector">→</span>')
    chunks.append("</div>")
    st.markdown("".join(chunks), unsafe_allow_html=True)
    if current:
        st.caption(f"Current stage: {STAGE_LABELS[current]}")


def _trace_event_signature(events: list[TraceEvent]) -> tuple[str, ...]:
    """Return a stable session-state signature for chronological trace events."""
    return tuple(event.event_id for event in events)


def _jump_to_latest_html() -> str:
    """Render a dependency-free browser anchor styled as a jump control."""
    return (
        '<a class="jump-latest" href="#trace-latest" '
        'aria-label="Jump to the latest workflow trace event">↓ Jump to latest</a>'
    )


def _render_trace(events: list[TraceEvent]) -> None:
    st.subheader("Live workflow trace")
    if not events:
        st.caption("Waiting for the first workflow event…")
        return

    current_signature = _trace_event_signature(events)
    previous_signature = st.session_state.get("trace_seen_event_ids", tuple())
    if previous_signature and current_signature != previous_signature:
        st.session_state["trace_has_new_events"] = True
    st.session_state["trace_seen_event_ids"] = current_signature

    toolbar_parts = ['<div class="trace-toolbar">']
    if st.session_state.get("trace_has_new_events"):
        toolbar_parts.append('<span class="trace-new-events">New trace events are available below.</span>')
    else:
        toolbar_parts.append('<span></span>')
    toolbar_parts.append(_jump_to_latest_html())
    toolbar_parts.append('</div>')
    st.markdown("".join(toolbar_parts), unsafe_allow_html=True)

    for event in events:
        st.markdown(
            f'<div class="trace-row">'
            f'<div class="trace-stage">{html.escape(event.stage.title())}</div>'
            f'<div class="trace-message">{html.escape(event.message)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div id="trace-latest"></div>', unsafe_allow_html=True)
    if st.session_state.get("trace_has_new_events"):
        if st.button("Mark as caught up", key="trace_caught_up", help="Hide the new-event notice until another trace event arrives."):
            st.session_state["trace_has_new_events"] = False
            st.rerun()


def _render_live_workspace(client: ResearchAPIClient) -> None:
    report_id = st.session_state.get("selected_report_id")
    topic = st.session_state.get("selected_topic") or "Research"

    if not report_id:
        st.session_state["workspace_mode"] = "landing"
        st.rerun()

    trace: TraceResponse | None = st.session_state.get("current_trace")
    if trace is None or trace.report_id != report_id:
        try:
            trace = client.get_trace(report_id)
            st.session_state["current_trace"] = trace
        except APIClientError as exc:
            st.error(exc.message)
            return

    st.markdown(
        f'<div class="live-header"><h1>{html.escape(topic)}</h1>'
        f'<div>Report ID: <code>{html.escape(report_id)}</code></div></div>',
        unsafe_allow_html=True,
    )

    if trace.events:
        attempts = sorted({event.attempt for event in trace.events})
        for attempt in attempts:
            _render_stepper(trace.events, attempt)

    status_text = trace.status.value.capitalize()
    st.status(status_text, state="complete" if trace.status is not ExecutionStatus.RUNNING else "running")

    _render_trace(trace.events)

    if trace.status is ExecutionStatus.COMPLETED:
        _complete_live_workspace(client, report_id)
    elif trace.status is ExecutionStatus.FAILED:
        st.error(
            "Research failed. No automatic retry was started.",
        )
        st.caption(f"Report ID: {report_id}")
        st.session_state["polling_active"] = False
    else:
        st.caption("Research is still running. The trace will refresh automatically.")


def _complete_live_workspace(client: ResearchAPIClient, report_id: str) -> None:
    st.session_state["polling_active"] = False

    report = st.session_state.get("current_report")

    if report is None:
        try:
            payload = client.get_report(report_id)
        except APIClientError as exc:
            st.session_state["report_error"] = exc.message
            st.warning("Research completed, but the final report could not be loaded.")

            if st.button("Retry", key=f"report_retry_{report_id}"):
                try:
                    payload = client.get_report(report_id)
                    report = payload["report"]
                    st.session_state["current_report"] = report
                    st.session_state["selected_quality"] = QualityLevel(report["quality"])
                    st.session_state["report_error"] = None
                    st.rerun()
                except APIClientError as retry_exc:
                    st.session_state["report_error"] = retry_exc.message
                    return

            return

        report = payload["report"]
        st.session_state["current_report"] = report
        st.session_state["selected_quality"] = QualityLevel(report["quality"])
        st.session_state["report_error"] = None
    else:
        st.session_state["selected_quality"] = QualityLevel(report["quality"])

    attempts = max(
        (event.attempt for event in st.session_state["current_trace"].events),
        default=1,
    )

    st.success(
        f"Research complete · {QualityLevel(report['quality']).value.capitalize()} quality · "
        f"{attempts} attempt{'s' if attempts != 1 else ''}"
    )
    _render_report(report)


def _citation_links_html(text: str, valid_citations: set[int]) -> str:
    """Escape report text and convert approved inline citations into anchors."""
    escaped = html.escape(text or "")

    def replace(match) -> str:
        citation_id = int(match.group(1))
        if citation_id not in valid_citations:
            return match.group(0)
        return (
            f'<a class="report-link" href="#source-{citation_id}" '
            f'aria-label="Open reference {citation_id}">[{citation_id}]</a>'
        )

    return re.sub(r"\[(\d+)\]", replace, escaped)


def _source_lookup(sources: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index report references by citation ID for safe citation rendering."""
    return {int(source["citation_id"]): source for source in sources}


def _render_report(report: dict[str, Any]) -> None:
    st.markdown('<div class="report-shell">', unsafe_allow_html=True)
    references = report.get("references", [])
    source_by_id = _source_lookup(references)
    valid_citations = set(source_by_id)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("Executive Summary")
    st.markdown(
        f'<div class="report-summary">{_citation_links_html(report.get("executive_summary", ""), valid_citations)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section" id="findings-section">', unsafe_allow_html=True)
    st.header("Key Findings")
    findings = report.get("key_findings", [])
    if not findings:
        st.markdown('<div class="report-empty">No key findings.</div>', unsafe_allow_html=True)
    for index, finding in enumerate(findings, start=1):
        citations = [int(value) for value in finding.get("supporting_sources", []) if int(value) in valid_citations]
        citation_html = " ".join(
            f'<a class="report-link" href="#source-{citation}" aria-label="Open reference {citation}">[{citation}]</a>'
            for citation in citations
        )
        badge = html.escape(str(finding.get("confidence", "")).capitalize())
        st.markdown(
            f'<div class="report-item" id="finding-{index}">'
            f'<div class="report-item-title">{index}. {html.escape(str(finding.get("claim", "")))}'
            f'<span class="confidence-badge">{badge}</span></div>'
            f'<div class="report-meta">Supporting sources: {citation_html or "—"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("Supporting Evidence")
    evidence_items = report.get("supporting_evidence", [])
    if not evidence_items:
        st.markdown('<div class="report-empty">No supporting evidence entries.</div>', unsafe_allow_html=True)
    for index, raw_evidence in enumerate(evidence_items, start=1):
        if isinstance(raw_evidence, str):
            evidence = {
                "text": raw_evidence,
                "citation_ids": [],
                "related_finding_indices": [],
            }
        else:
            evidence = raw_evidence

        citation_ids = [
            int(value) for value in evidence.get("citation_ids", []) if int(value) in valid_citations
        ]
        citation_html = " ".join(
            f'<a class="report-link" href="#source-{citation}" aria-label="Open reference {citation}">[{citation}]</a>'
            for citation in citation_ids
        )
        relationship_html = " ".join(
            f'<a class="relation-link" href="#finding-{int(finding_index)}">Supports Finding {int(finding_index)}</a>'
            for finding_index in evidence.get("related_finding_indices", [])
            if 1 <= int(finding_index) <= len(findings)
        )
        body = _citation_links_html(str(evidence.get("text", "")), valid_citations)
        metadata = relationship_html or "<span class=\"report-empty\">Contextual evidence</span>"
        if citation_html:
            metadata = f"{metadata}<br>Sources: {citation_html}"
        st.markdown(
            f'<div class="report-item">'
            f'<div class="report-item-title">{index}. Supporting evidence</div>'
            f'<div class="report-body">{body}</div>'
            f'<div class="report-meta">{metadata}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("Gaps")
    gaps = report.get("gaps", [])
    gap_explanations = report.get("gap_explanations", [])
    if not gaps:
        st.markdown('<div class="report-empty">No identified gaps.</div>', unsafe_allow_html=True)
    for index, gap in enumerate(gaps, start=1):
        explanation = gap_explanations[index - 1] if index - 1 < len(gap_explanations) else ""
        gap_type = html.escape(str(gap.get("type", "")).replace("_", " ").capitalize())
        st.markdown(
            f'<div class="report-item">'
            f'<div class="report-item-title">{index}. {html.escape(str(gap.get("description", "")))}'
            f'<span class="type-badge">{gap_type}</span></div>'
            f'<div class="report-body">{html.escape(str(explanation))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("Conflicts")
    conflicts = report.get("conflicts", [])
    conflict_explanations = report.get("conflict_explanations", [])
    if not conflicts:
        st.markdown('<div class="report-empty">No identified conflicts.</div>', unsafe_allow_html=True)
    for index, conflict in enumerate(conflicts, start=1):
        explanation = conflict_explanations[index - 1] if index - 1 < len(conflict_explanations) else ""
        related_ids = [int(value) for value in conflict.get("related_sources", []) if int(value) in valid_citations]
        citation_html = " ".join(
            f'<a class="report-link" href="#source-{citation}" aria-label="Open reference {citation}">[{citation}]</a>'
            for citation in related_ids
        )
        st.markdown(
            f'<div class="report-item">'
            f'<div class="report-item-title">{index}. {html.escape(str(conflict.get("description", "")))}</div>'
            f'<div class="report-body">{html.escape(str(explanation))}</div>'
            f'<div class="report-meta">Related sources: {citation_html or "—"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.header("References")
    if not references:
        st.markdown('<div class="report-empty">No references.</div>', unsafe_allow_html=True)
    for source in references:
        citation_id = int(source["citation_id"])
        title = html.escape(str(source.get("title", "Untitled source")))
        url = html.escape(str(source.get("url", "")), quote=True)
        st.markdown(
            f'<div class="source-card" id="source-{citation_id}">'
            f'<span class="source-number">{citation_id}.</span> '
            f'<a class="report-link" href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def _refresh_after_terminal_trace(client: ResearchAPIClient, status: ExecutionStatus) -> None:
    """Refresh sidebar state after an execution reaches a terminal status."""
    _refresh_active(client)

    if status is ExecutionStatus.COMPLETED:
        _refresh_history(client)


def _handle_terminal_trace_refresh(
    client: ResearchAPIClient,
    status: ExecutionStatus,
) -> None:
    """Refresh sidebar state and request a full app rerun after termination."""
    _refresh_after_terminal_trace(client, status)
    st.rerun()


@st.fragment(run_every=POLL_INTERVAL_SECONDS)
def _live_poll_fragment(client: ResearchAPIClient) -> None:
    if st.session_state.get("workspace_mode") != "live":
        return

    report_id = st.session_state.get("selected_report_id")
    if not report_id:
        return

    if st.session_state.get("polling_active"):
        try:
            trace = client.get_trace(report_id)
            st.session_state["current_trace"] = trace
        except APIClientError as exc:
            st.session_state["execution_error"] = exc.message
            trace = st.session_state.get("current_trace")
        else:
            if trace.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
                st.session_state["polling_active"] = False
                _handle_terminal_trace_refresh(client, trace.status)

    _render_live_workspace(client)


def main() -> None:
    _configure_page()
    _initialize_state()
    client = ResearchAPIClient()
    _ensure_history(client)
    _refresh_active(client)
    _render_sidebar(client)

    mode = st.session_state["workspace_mode"]
    if mode == "landing":
        _render_landing(client)
    elif mode == "live":
        _live_poll_fragment(client)
    elif mode == "history":
        if st.button("New Research", key="new_research"):
            st.session_state["workspace_mode"] = "landing"
            st.session_state["selected_report_id"] = None
            st.session_state["selected_topic"] = None
            st.session_state["current_report"] = None
            st.session_state["current_trace"] = None
            st.session_state["report_error"] = None
            st.session_state["execution_error"] = None
            st.rerun()
        if st.session_state.get("report_error"):
            st.error(st.session_state["report_error"])
        report = st.session_state.get("current_report")
        if report:
            _render_report(report)


if __name__ == "__main__":
    main()
