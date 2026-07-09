"""Hub persistence layer (v3 Phase B, SB-B1).

SQLite-backed store for research projects — the persistence half of the hub's
core. Pure Python, no UI knowledge, so the Streamlit view (SB-B3) or any later
frontend is a thin caller. State lives in SQLite (never st.session_state), so
projects survive a restart — that's the daily-use acceptance criterion.

A project's `domain` is the name of a registered DomainProfile; creation and
updates validate against the profile registry so the picker can't persist a
domain the pipeline can't load.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.domain import available

DEFAULT_DB_PATH = Path("data/hub.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    domain     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class Project:
    """Immutable snapshot of a row in `projects`."""

    id: int
    name: str
    domain: str
    created_at: str


class HubStore:
    """SQLite CRUD for research projects. Returns immutable `Project` snapshots."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_project(self, name: str, domain: str) -> Project:
        name = name.strip()
        if not name:
            raise ValueError("project name must be non-empty")
        self._require_known_domain(domain)
        cur = self._conn.execute(
            "INSERT INTO projects (name, domain) VALUES (?, ?)", (name, domain)
        )
        self._conn.commit()
        return self.get_project(cur.lastrowid)

    def get_project(self, project_id: int) -> Project:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no project with id {project_id}")
        return _row_to_project(row)

    def list_projects(self) -> list[Project]:
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [_row_to_project(r) for r in rows]

    def set_project_domain(self, project_id: int, domain: str) -> Project:
        self._require_known_domain(domain)
        self.get_project(project_id)  # raises KeyError if missing
        self._conn.execute(
            "UPDATE projects SET domain = ? WHERE id = ?", (domain, project_id)
        )
        self._conn.commit()
        return self.get_project(project_id)

    @staticmethod
    def _require_known_domain(domain: str) -> None:
        if domain not in available():
            raise ValueError(f"unknown domain {domain!r}; registered: {available()}")


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        domain=row["domain"],
        created_at=row["created_at"],
    )
