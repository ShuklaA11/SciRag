"""DomainProfile: declarative config for domain-adaptive SciRAG (v3 Phase A).

Everything that is domain-coupled — the section taxonomy and its head-matching
rules, the embedder, the verification strategy, the field's canonical data
sources — is captured here so the pipeline can be retargeted to a new research
domain by swapping profiles instead of editing code.

Active profile is selected by env var, mirroring the LLM provider pattern
(``SCIRAG_LLM_PROVIDER``): ``SCIRAG_DOMAIN=nlp_ml`` (default). Built-in profiles
register themselves on import of ``src.domain`` — see ``profiles.py``.

This module is pure config + registry; it does not import any pipeline code,
so consumers (chunker, router, verifier) depend on it, not the reverse.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

DEFAULT_DOMAIN_ENV = "SCIRAG_DOMAIN"
DEFAULT_DOMAIN = "nlp_ml"


@dataclass(frozen=True)
class DomainProfile:
    """Immutable, declarative description of one research domain.

    ``section_patterns`` is an ordered tuple of ``(label, raw_regex)`` — order
    matters (more-specific patterns first), matching the legacy behavior in
    ``section_chunker._SECTION_PATTERNS``. Every label must appear in
    ``section_types``; ``section_types`` also holds the fallback bucket (e.g.
    ``"other"``) which has no pattern.
    """

    name: str
    section_types: tuple[str, ...]
    section_patterns: tuple[tuple[str, str], ...]
    embedder_name: str
    verification_strategy: str  # "nli" | "numeric" | "none"
    verification_model: str
    data_sources: tuple[str, ...] = field(default_factory=tuple)
    eval_benchmark: str = ""

    def __post_init__(self) -> None:
        labels = {label for label, _ in self.section_patterns}
        unknown = labels - set(self.section_types)
        if unknown:
            raise ValueError(
                f"profile {self.name!r}: pattern labels {sorted(unknown)} "
                f"not in section_types"
            )

    def compiled_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Compile ``section_patterns`` to ``(label, Pattern)`` in order.

        Case-insensitive, matching the legacy chunker. Consumers should build
        their matching table from this rather than hardcoding regexes.
        """
        return [(label, re.compile(rx, re.IGNORECASE)) for label, rx in self.section_patterns]


# --- registry ---------------------------------------------------------------

_REGISTRY: dict[str, DomainProfile] = {}


def register(profile: DomainProfile) -> DomainProfile:
    """Register a profile by name (idempotent overwrite). Returns it."""
    _REGISTRY[profile.name] = profile
    return profile


def get_profile(name: str) -> DomainProfile:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown domain profile {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def active_profile() -> DomainProfile:
    """The profile selected by ``$SCIRAG_DOMAIN`` (default ``nlp_ml``)."""
    return get_profile(os.getenv(DEFAULT_DOMAIN_ENV, DEFAULT_DOMAIN))
