"""histo-omics-lite: lean multimodal histology + transcriptomics toolkit."""

from __future__ import annotations

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover - fallback during development
    __version__ = "0.0.0"

__all__ = ["__version__"]
