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

import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.domain import available

DEFAULT_DB_PATH = Path("data/hub.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    domain     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id),
    idea        TEXT NOT NULL,
    git_commit  TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def current_git_commit() -> str:
    """Short HEAD sha, ``"unknown"`` outside a repo. Mirrors the scripts'
    ``_git_commit`` provenance instinct so eval runs and stored reports share
    one commit-stamping convention."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Project:
    """Immutable snapshot of a row in `projects`."""

    id: int
    name: str
    domain: str
    created_at: str


@dataclass(frozen=True)
class Evaluation:
    """Immutable snapshot of a row in `evaluations`. ``report`` is the parsed
    IdeaReport JSON (a dict), stamped with the ``git_commit`` at save time."""

    id: int
    project_id: int
    idea: str
    git_commit: str
    report: dict[str, Any]
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

    def save_evaluation(
        self, project_id: int, report: Any, *, git_commit: str | None = None
    ) -> Evaluation:
        """Persist an ``IdeaReport`` (any dataclass) under a project.

        Serialized structurally via ``asdict`` — the store stays decoupled from
        ``src.ideas``. ``git_commit`` defaults to the current HEAD sha.
        """
        self.get_project(project_id)  # raises KeyError if missing
        payload = asdict(report)
        idea = payload.get("idea", "")
        commit = git_commit if git_commit is not None else current_git_commit()
        cur = self._conn.execute(
            "INSERT INTO evaluations (project_id, idea, git_commit, report_json) "
            "VALUES (?, ?, ?, ?)",
            (project_id, idea, commit, json.dumps(payload, default=str)),
        )
        self._conn.commit()
        return self.get_evaluation(cur.lastrowid)

    def get_evaluation(self, evaluation_id: int) -> Evaluation:
        row = self._conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no evaluation with id {evaluation_id}")
        return _row_to_evaluation(row)

    def list_evaluations(self, project_id: int) -> list[Evaluation]:
        rows = self._conn.execute(
            "SELECT * FROM evaluations WHERE project_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (project_id,),
        ).fetchall()
        return [_row_to_evaluation(r) for r in rows]

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


def _row_to_evaluation(row: sqlite3.Row) -> Evaluation:
    return Evaluation(
        id=row["id"],
        project_id=row["project_id"],
        idea=row["idea"],
        git_commit=row["git_commit"],
        report=json.loads(row["report_json"]),
        created_at=row["created_at"],
    )
