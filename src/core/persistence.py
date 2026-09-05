"""SQLite-backed report, memory, and execution-trace persistence."""

from __future__ import annotations

import json
import sqlite3
import string
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.models.schemas import (
    FinalReport,
    MemoryContext,
    MemoryMatch,
    QualityLevel,
    StoredReport,
)

DEFAULT_DB_PATH = Path("data") / "research_agent.db"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "will", "would", "about", "into", "than",
    "then", "these", "those", "through", "using", "use", "used", "can", "could",
}


class PersistenceError(RuntimeError):
    """Raised when a persistence operation cannot be completed."""


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    report_id: str
    timestamp: datetime
    stage: str
    event_type: str
    message: str
    attempt: int = 0


class SQLitePersistence:
    """Persist final reports and human-readable stage-level execution events."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reports (
                        report_id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        quality TEXT NOT NULL,
                        report_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_created_at "
                    "ON reports(created_at)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_quality "
                    "ON reports(quality)"
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS execution_events (
                        event_id TEXT PRIMARY KEY,
                        report_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(report_id) REFERENCES reports(report_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_report_time "
                    "ON execution_events(report_id, timestamp)"
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to initialize SQLite database: {exc}") from exc

    def create_report(self, topic: str) -> str:
        topic = topic.strip()
        if not topic:
            raise ValueError("Research topic cannot be blank")

        report_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        placeholder = json.dumps({"topic": topic})
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO reports(report_id, topic, created_at, quality, report_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (report_id, topic, created_at.isoformat(), "pending", placeholder),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to create report record: {exc}") from exc
        return report_id

    def save_report(
        self,
        report_id: str,
        report: FinalReport,
        *,
        created_at: datetime | None = None,
    ) -> StoredReport:
        if report.quality is None:
            raise PersistenceError(
                "A final report must have a quality level before persistence"
            )

        if created_at is None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT created_at FROM reports WHERE report_id = ?", (report_id,)
                ).fetchone()
            if row is None:
                raise PersistenceError(f"Unknown report_id: {report_id}")
            timestamp = datetime.fromisoformat(row["created_at"])
        else:
            timestamp = created_at
        stored = StoredReport(
            report_id=report_id,
            topic=report.topic,
            created_at=timestamp,
            quality=report.quality,
            report=report,
        )
        payload = json.dumps(report.model_dump(mode="json"))
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE reports
                    SET topic = ?, created_at = ?, quality = ?, report_json = ?
                    WHERE report_id = ?
                    """,
                    (
                        stored.topic,
                        stored.created_at.isoformat(),
                        stored.quality.value,
                        payload,
                        report_id,
                    ),
                )
                if connection.execute(
                    "SELECT changes()"
                ).fetchone()[0] != 1:
                    raise PersistenceError(f"Unknown report_id: {report_id}")
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to save report {report_id}: {exc}") from exc
        return stored

    def get_report(self, report_id: str) -> StoredReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE report_id = ? AND quality != 'pending'",
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            report = FinalReport.model_validate(json.loads(row["report_json"]))
            return StoredReport(
                report_id=row["report_id"],
                topic=row["topic"],
                created_at=datetime.fromisoformat(row["created_at"]),
                quality=QualityLevel(row["quality"]),
                report=report,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError(f"Stored report {report_id} is invalid: {exc}") from exc

    def list_reports(self, limit: int = 50) -> list[StoredReport]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM reports
                WHERE quality != 'pending'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_stored_report(row) for row in rows]

    def _row_to_stored_report(self, row: sqlite3.Row) -> StoredReport:
        try:
            report = FinalReport.model_validate(json.loads(row["report_json"]))
            return StoredReport(
                report_id=row["report_id"],
                topic=row["topic"],
                created_at=datetime.fromisoformat(row["created_at"]),
                quality=QualityLevel(row["quality"]),
                report=report,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Stored report {row['report_id']} is invalid: {exc}"
            ) from exc

    def export_report_json(self, report_id: str) -> str:
        stored = self.get_report(report_id)
        if stored is None:
            raise PersistenceError(f"Unknown or unfinished report_id: {report_id}")
        return json.dumps(stored.report.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def record_event(
        self,
        report_id: str,
        *,
        stage: str,
        event_type: str,
        message: str,
        attempt: int = 0,
        timestamp: datetime | None = None,
    ) -> ExecutionEvent:
        if not stage.strip() or not event_type.strip() or not message.strip():
            raise ValueError("Trace stage, event_type, and message cannot be blank")
        if attempt < 0:
            raise ValueError("Trace attempt cannot be negative")

        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            report_id=report_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            stage=stage.strip(),
            event_type=event_type.strip(),
            message=message.strip(),
            attempt=attempt,
        )
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM reports WHERE report_id = ?", (report_id,)
                ).fetchone() is None:
                    raise PersistenceError(f"Unknown report_id: {report_id}")
                connection.execute(
                    """
                    INSERT INTO execution_events(
                        event_id, report_id, timestamp, stage, event_type, message, attempt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.report_id,
                        event.timestamp.isoformat(),
                        event.stage,
                        event.event_type,
                        event.message,
                        event.attempt,
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to record execution event: {exc}") from exc
        return event

    def get_trace(self, report_id: str) -> list[ExecutionEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_events
                WHERE report_id = ?
                ORDER BY timestamp ASC, rowid ASC
                """,
                (report_id,),
            ).fetchall()
        return [
            ExecutionEvent(
                event_id=row["event_id"],
                report_id=row["report_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                stage=row["stage"],
                event_type=row["event_type"],
                message=row["message"],
                attempt=row["attempt"],
            )
            for row in rows
        ]

    def find_memory(self, topic: str, limit: int = 2) -> MemoryContext:
        if limit <= 0:
            return MemoryContext(matches=[])
        query_tokens = _meaningful_tokens(topic)
        if not query_tokens:
            return MemoryContext(matches=[])

        ranked: list[tuple[int, StoredReport]] = []
        for stored in self.list_reports(limit=10_000):
            overlap = len(query_tokens & _report_tokens(stored.report))
            if overlap >= 2:
                ranked.append((overlap, stored))

        ranked.sort(
            key=lambda item: (-item[0], -item[1].created_at.timestamp())
        )
        return MemoryContext(
            matches=[
                MemoryMatch(
                    report_id=stored.report_id,
                    topic=stored.topic,
                    key_findings=stored.report.key_findings,
                    gaps=stored.report.gaps,
                    conflicts=stored.report.conflicts,
                )
                for _, stored in ranked[:limit]
            ]
        )


def _meaningful_tokens(text: str) -> set[str]:
    table = str.maketrans("", "", string.punctuation)
    return {
        token
        for token in text.translate(table).lower().split()
        if token and token not in _STOPWORDS
    }


def _report_tokens(report: FinalReport) -> set[str]:
    # Memory matching is deliberately topic-based. Prior report findings and
    # gaps are returned as context only; they are not used to manufacture
    # relevance from arbitrary content overlap.
    return _meaningful_tokens(report.topic)
