"""Light-curve transit photometry feature extraction."""

from __future__ import annotations

from importlib.metadata import version as _metadata_version

__version__ = _metadata_version("shaula")

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
