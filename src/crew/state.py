"""Shared state for the research-crew graph (SB-X2).

The crew is a supervisor loop over a shared scratchpad. ``findings`` uses an
additive reducer so each specialist *appends* its ToolResult (never clobbers
the others); the remaining fields are last-write-wins and are only written by
the supervisor / synthesis nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.crew.tools import ToolResult

# The specialist agents the supervisor can route to, plus the terminal signal.
AGENTS: tuple[str, ...] = ("search", "verify", "novelty")
FINISH = "FINISH"


class CrewState(TypedDict, total=False):
    query: str
    findings: Annotated[list[ToolResult], operator.add]
    next_agent: str
    iterations: int
    answer: str
