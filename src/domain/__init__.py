"""Domain-adaptive configuration for SciRAG (v3 Phase A).

Importing this package registers all built-in profiles, so
``from src.domain import active_profile`` works out of the box.
"""

from src.domain.profile import (
    DomainProfile,
    active_profile,
    available,
    get_profile,
    register,
)
from src.domain import profiles as _profiles  # noqa: F401  (self-registers built-ins)

__all__ = [
    "DomainProfile",
    "active_profile",
    "available",
    "get_profile",
    "register",
]
