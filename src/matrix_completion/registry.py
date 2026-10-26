"""Extensible registry mapping public method names to factorization functions."""

from collections.abc import Callable

from .backend import BackendResult

Method = Callable[..., BackendResult]
_METHODS: dict[str, Method] = {}
_ALIASES: dict[str, str] = {}


def register_method(name: str, function: Method, aliases=(), *, replace=False):
    """Register a completion backend and optional backward-compatible aliases."""
    normalized = _normalize(name)
    if normalized in _METHODS and not replace:
        raise ValueError(f"method {normalized!r} is already registered")
    _METHODS[normalized] = function
    for alias in aliases:
        _ALIASES[_normalize(alias)] = normalized


def resolve_method(name: str):
    normalized = _normalize(name)
    canonical = _ALIASES.get(normalized, normalized)
    if canonical not in _METHODS:
        choices = ", ".join(available_methods())
        raise ValueError(f"unknown method {name!r}; available methods: {choices}")
    return canonical, _METHODS[canonical]


def available_methods():
    """Return canonical built-in and user-registered method names."""
    return tuple(sorted(_METHODS))


def _normalize(name):
    return str(name).strip().lower().replace("-", "_")
