"""Light-curve transit photometry feature extraction."""

from __future__ import annotations

__version__ = "0.1.0"

from .api import (  # noqa: E402  (version must be defined before api imports it)
    STAGES,
    ExtractionResult,
    ProgressCallback,
    ProgressEvent,
    extract,
)

__all__ = [
    "STAGES",
    "ExtractionResult",
    "ProgressCallback",
    "ProgressEvent",
    "__version__",
    "extract",
]
