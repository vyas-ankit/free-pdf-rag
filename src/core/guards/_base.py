# src/core/guards/_base.py
"""GuardResult dataclass — every guard returns one of these."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuardResult:
    """Outcome of a single guard, or of the full input-guards pipeline.

    Fields:
        passed:   True if safe to continue, False if the guard rejected.
        reason:   Short machine-readable code (e.g. "length_exceeded",
                  "injection_keyword"). Used for logging/metrics.
        message:  User-facing rejection message (None if passed).
        details:  Arbitrary debug context (matched pattern, score, etc.).
    """
    passed: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    details: dict = field(default_factory=dict)
