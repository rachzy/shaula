"""Light-curve transit photometry feature extraction."""

from __future__ import annotations

__version__ = "0.0.1a0"

from .api import (  # noqa: E402  (version must be defined before api imports it)
    STAGES,
    ExtractionResult,
    ProgressCallback,
    ProgressEvent,
    ResolvedTarget,
    TargetNotFound,
    extract,
    resolve,
)

__all__ = [
    "STAGES",
    "ExtractionResult",
    "ProgressCallback",
    "ProgressEvent",
    "ResolvedTarget",
    "TargetNotFound",
    "__version__",
    "extract",
    "resolve",
]
