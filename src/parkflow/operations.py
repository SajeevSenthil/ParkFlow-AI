"""Operator workflow persistence (PRD section 9.2) — turns the read-only dashboard
into an operations tool a command centre can actually *act* in.

A reporting tool shows charts; an operations tool lets an operator acknowledge a
deployment, override an assignment, and mark it complete — and remembers all of it.
That state lives in a small SQLite log (Python stdlib ``sqlite3`` — no new dependency),
kept deliberately *separate* from the precomputed ML artifacts so the "dashboard only
reads artifacts" boundary still holds for everything the model produces.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

log = get_logger("operations")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    team TEXT NOT NULL,
    zone TEXT NOT NULL,
    priority_score REAL,
    predicted_violations REAL,
    status TEXT NOT NULL,
    operator TEXT,
    note TEXT
);
"""

_COLUMNS = [
    "id", "ts", "team", "zone", "priority_score",
    "predicted_violations", "status", "operator", "note",
]


def init_db(path: str | Path) -> Path:
    """Create the deployments table if it does not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(_SCHEMA)
    return path


def log_deployment(
    path: str | Path,
    team: str,
    zone: str,
    priority_score: float | None = None,
    predicted_violations: float | None = None,
    status: str = "deployed",
    operator: str | None = None,
    note: str | None = None,
) -> int:
    """Record a deployment action; returns the new row id."""
    init_db(path)
    with sqlite3.connect(path) as con:
        cur = con.execute(
            "INSERT INTO deployments "
            "(ts, team, zone, priority_score, predicted_violations, status, operator, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                team, zone, priority_score, predicted_violations, status, operator, note,
            ),
        )
        new_id = int(cur.lastrowid)
    log.info("Logged deployment #%d: %s -> %s (%s)", new_id, team, zone, status)
    return new_id


def override_assignment(
    path: str | Path,
    team: str,
    new_zone: str,
    priority_score: float | None = None,
    operator: str | None = None,
    note: str | None = None,
) -> int:
    """Reassign a team to a different zone (logged with status='overridden')."""
    return log_deployment(
        path, team, new_zone, priority_score=priority_score,
        status="overridden", operator=operator, note=note or "manual override",
    )


def mark_complete(path: str | Path, deployment_id: int) -> None:
    """Transition a deployment to status='completed'."""
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE deployments SET status = 'completed' WHERE id = ?", (deployment_id,)
        )
    log.info("Marked deployment #%d complete", deployment_id)


def deployment_history(path: str | Path, limit: int = 100) -> pd.DataFrame:
    """Most-recent-first deployment log (empty frame if the DB does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(
            "SELECT * FROM deployments ORDER BY id DESC LIMIT ?", con, params=(limit,)
        )
