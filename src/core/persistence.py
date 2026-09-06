"""SQLite-backed report, memory, and execution-trace persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import string
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.models.schemas import FinalReport, MemoryContext, MemoryMatch, QualityLevel, StoredReport

logger = logging.getLogger(__name__)
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
    attempt: int


class SQLitePersistence:
    """Persist completed reports and stage-level execution history in SQLite."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.db_path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            return connection
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to connect to SQLite database: {exc}") from exc

    def _initialize(self) -> None:
        try:
            # WAL is configured once during database initialization so concurrent
            # trace readers can coexist with workflow writers. Per-connection
            # busy_timeout remains configured in _connect().
            with sqlite3.connect(self.db_path, timeout=5.0) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA busy_timeout = 5000")
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reports (
                        report_id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        quality TEXT NOT NULL,
                        report_json TEXT NOT NULL,
                        execution_status TEXT NOT NULL DEFAULT 'running'
                    )
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(reports)").fetchall()
                }
                if "execution_status" not in columns:
                    connection.execute(
                        "ALTER TABLE reports ADD COLUMN execution_status TEXT NOT NULL DEFAULT 'running'"
                    )
                    connection.execute(
                        "UPDATE reports SET execution_status = CASE "
                        "WHEN quality = 'pending' THEN 'running' ELSE 'completed' END"
                    )

                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_quality ON reports(quality)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(execution_status)"
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
                        attempt INTEGER NOT NULL,
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
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO reports(
                        report_id, topic, created_at, quality, report_json, execution_status
                    ) VALUES (?, ?, ?, ?, ?, 'running')
                    """,
                    (report_id, topic, created_at.isoformat(), "pending", json.dumps({"topic": topic})),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to create report record: {exc}") from exc
        return report_id

    def save_report(self, report_id: str, report: FinalReport) -> StoredReport:
        if report.quality is None:
            raise PersistenceError("A final report must have a quality level before persistence")

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT created_at FROM reports WHERE report_id = ?", (report_id,)
                ).fetchone()
                if row is None:
                    raise PersistenceError(f"Unknown report_id: {report_id}")

                timestamp = datetime.fromisoformat(row["created_at"])
                stored = StoredReport(
                    report_id=report_id,
                    topic=report.topic,
                    created_at=timestamp,
                    quality=report.quality,
                    report=report,
                )
                payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)

                # Terminal report persistence and completion status are atomic.
                connection.execute("BEGIN")
                cursor = connection.execute(
                    """
                    UPDATE reports
                    SET topic = ?, created_at = ?, quality = ?, report_json = ?, execution_status = 'completed'
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
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise PersistenceError(f"Unknown report_id: {report_id}")
                connection.commit()
                return stored
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to save report {report_id}: {exc}") from exc

    def mark_failed(self, report_id: str) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE reports SET execution_status = 'failed' WHERE report_id = ?",
                    (report_id,),
                )
                if cursor.rowcount != 1:
                    raise PersistenceError(f"Unknown report_id: {report_id}")
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to mark report {report_id} as failed: {exc}") from exc

    def get_execution_status(self, report_id: str) -> str | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT execution_status FROM reports WHERE report_id = ?", (report_id,)
                ).fetchone()
            return None if row is None else str(row["execution_status"])
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to read execution status: {exc}") from exc

    def get_report(self, report_id: str) -> StoredReport | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM reports WHERE report_id = ? AND execution_status = 'completed'",
                    (report_id,),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_stored_report(row)
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to read report {report_id}: {exc}") from exc

    def list_reports(self, limit: int = 20) -> list[StoredReport]:
        if limit <= 0:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM reports
                    WHERE execution_status = 'completed'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [self._row_to_stored_report(row) for row in rows]
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to list reports: {exc}") from exc

    @staticmethod
    def _row_to_stored_report(row: sqlite3.Row) -> StoredReport:
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
            raise PersistenceError(f"Stored report {row['report_id']} is invalid: {exc}") from exc

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
        attempt: int = 1,
        timestamp: datetime | None = None,
    ) -> ExecutionEvent:
        if not stage.strip() or not event_type.strip() or not message.strip():
            raise ValueError("Trace stage, event_type, and message cannot be blank")
        if attempt < 1:
            raise ValueError("Trace attempt must be at least 1")

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
        except PersistenceError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to record execution event: {exc}") from exc
        return event

    def get_trace(self, report_id: str) -> list[ExecutionEvent]:
        try:
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
        except sqlite3.Error as exc:
            raise PersistenceError(f"Unable to read trace: {exc}") from exc

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

        ranked.sort(key=lambda item: (-item[0], -item[1].created_at.timestamp()))
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
        token for token in text.translate(table).lower().split()
        if token and token not in _STOPWORDS
    }


def _report_tokens(report: FinalReport) -> set[str]:
    return _meaningful_tokens(report.topic)
