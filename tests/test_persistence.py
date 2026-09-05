from datetime import datetime, timezone

import pytest

from src.core.persistence import PersistenceError, SQLitePersistence
from src.models.schemas import (
    ConfidenceLevel,
    FinalReport,
    Finding,
    QualityLevel,
)


def make_report(topic: str) -> FinalReport:
    source = {
        "citation_id": 1,
        "title": "Example source",
        "url": "https://example.com/source",
        "retrieved_at": datetime.now(timezone.utc),
        "snippet": "Example evidence",
        "search_queries": [topic],
        "content": "Example content",
    }
    return FinalReport(
        topic=topic,
        executive_summary="Summary",
        key_findings=[
            Finding(
                claim="Battery storage deployment is growing",
                supporting_sources=[1],
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        supporting_evidence=["Battery storage deployment is growing [1]"],
        gaps=[],
        gap_explanations=[],
        conflicts=[],
        conflict_explanations=[],
        quality=QualityLevel.HIGH,
        references=[source],
        analysis_issues=[],
        unresolved_issues=[],
    )


def test_create_save_get_round_trip(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    report_id = db.create_report("Battery energy storage")
    saved = db.save_report(report_id, make_report("Battery energy storage"))

    loaded = db.get_report(report_id)
    assert loaded is not None
    assert loaded.report == saved.report
    assert loaded.report_id == report_id
    assert loaded.quality is QualityLevel.HIGH


def test_trace_is_stored_and_ordered(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    report_id = db.create_report("AI research")
    first = db.record_event(
        report_id,
        stage="research",
        event_type="research_started",
        message="Research started",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second = db.record_event(
        report_id,
        stage="critic",
        event_type="critic_review_completed",
        message="Critic verdict: PASS",
        attempt=1,
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    trace = db.get_trace(report_id)
    assert [event.event_id for event in trace] == [first.event_id, second.event_id]
    assert trace[1].attempt == 1


def test_trace_rejects_unknown_report(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    with pytest.raises(PersistenceError):
        db.record_event(
            "missing",
            stage="research",
            event_type="research_started",
            message="Research started",
        )


def test_memory_requires_two_meaningful_shared_tokens(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    id1 = db.create_report("Solar energy storage")
    db.save_report(id1, make_report("Solar energy storage"))

    assert len(db.find_memory("energy storage economics").matches) == 1
    assert db.find_memory("energy policy").matches == []


def test_memory_ranks_by_overlap_and_limits_to_two(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    for topic in ["AI battery storage", "Battery storage economics", "Battery storage policy"]:
        rid = db.create_report(topic)
        db.save_report(rid, make_report(topic))

    matches = db.find_memory("battery storage economics renewable").matches
    assert len(matches) == 2
    assert matches[0].topic == "Battery storage economics"
    assert matches[1].topic == "Battery storage policy" or matches[1].topic == "AI battery storage"


def test_json_export_returns_report_payload(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    rid = db.create_report("Climate policy")
    db.save_report(rid, make_report("Climate policy"))
    payload = db.export_report_json(rid)
    assert '"topic": "Climate policy"' in payload


def test_pending_reports_do_not_appear_in_history_or_memory(tmp_path):
    db = SQLitePersistence(tmp_path / "research_agent.db")
    db.create_report("Unfinished topic")
    assert db.list_reports() == []
    assert db.find_memory("Unfinished topic").matches == []
