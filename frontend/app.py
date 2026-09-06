"""Streamlit entry point for the Autonomous Research & Report Agent."""
from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import ResearchAPIClient
from frontend.models import APIClientError, QualityLevel


LANDING_COPY = (
    "Autonomous Research & Report Agent — Enter a research topic and let a "
    "multi-agent workflow research the web, analyze evidence, write a cited "
    "report, and critique it for gaps, conflicts, and quality."
)


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
        .stApp {
            background: #ffffff;
        }

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

        .sidebar-topic {
            font-weight: 600;
            color: #111827;
            line-height: 1.35;
            overflow: hidden;
        }

        .sidebar-meta {
            color: #6b7280;
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _quality_label(value: QualityLevel) -> str:
    return value.value.capitalize()


def _load_history(client: ResearchAPIClient) -> None:
    if st.session_state.get("history_loaded"):
        return

    with st.sidebar:
        st.caption("Loading history…")

    try:
        history = client.get_history()
    except APIClientError as exc:
        st.session_state["history_error"] = exc.message
        st.session_state["history_loaded"] = True
        return

    st.session_state["history"] = history
    st.session_state["history_loaded"] = True
    st.session_state["history_error"] = None


def _render_history(client: ResearchAPIClient) -> None:
    with st.sidebar:
        st.subheader("Research History")

        history_error = st.session_state.get("history_error")
        if history_error:
            st.error(history_error)
            if st.button("Retry", key="history_retry"):
                st.session_state["history_loaded"] = False
                st.rerun()

        history = st.session_state.get("history", [])
        if not history and not history_error:
            st.caption("No completed research yet.")

        for item in history:
            label = item.topic
            if st.button(
                label,
                key=f"history_{item.report_id}",
                use_container_width=True,
                help=label,
            ):
                st.session_state["selected_report_id"] = item.report_id
                st.session_state["selected_topic"] = item.topic
                st.session_state["workspace_mode"] = "history"
                st.rerun()

            st.markdown(
                f'<div class="sidebar-meta">{_quality_label(item.quality)} · '
                f'{item.created_at.astimezone().strftime("%d %b %Y")}</div>',
                unsafe_allow_html=True,
            )


def _render_landing() -> None:
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
        use_container_width=False,
        disabled=not topic.strip(),
    ):
        st.session_state["workspace_mode"] = "live"
        st.session_state["pending_topic"] = topic.strip()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_placeholder_workspace() -> None:
    topic = st.session_state.get("pending_topic") or st.session_state.get(
        "selected_topic", "Research"
    )
    st.title(topic)
    st.info(
        "The live execution workspace will be connected in the next frontend "
        "phase. The API client and page shell are ready."
    )


def main() -> None:
    _configure_page()
    client = ResearchAPIClient()

    _load_history(client)
    _render_history(client)

    mode = st.session_state.get("workspace_mode", "landing")
    if mode == "landing":
        _render_landing()
    else:
        if st.button("New Research", key="new_research"):
            st.session_state.clear()
            st.rerun()
        _render_placeholder_workspace()


if __name__ == "__main__":
    main()
