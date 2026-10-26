"""Mask-aware low-rank matrix completion with a unified API."""

from .api import MatrixCompleter
from .result import CompletionResult

__all__ = ["CompletionResult", "MatrixCompleter"]
__version__ = "0.1.0"
