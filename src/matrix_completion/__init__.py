"""Mask-aware low-rank matrix completion with a unified API."""

from .api import MatrixCompleter
from .backend import BackendResult
from .registry import available_methods, register_method
from .result import CompletionResult

__all__ = ["BackendResult", "CompletionResult", "MatrixCompleter", "available_methods",
           "register_method"]
__version__ = "0.1.0"
