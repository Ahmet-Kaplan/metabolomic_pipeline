"""Multiverse method registry — names → callables, self-registration on import.

A ``MethodSpec`` describes one method (e.g. half-minimum imputation): its
display name, citation, the category it belongs to (imputation / drift /
normalisation / transformation), and the callable that performs the work.

Each method module calls ``register(MethodSpec(...))`` at the bottom of its
file. The package ``methods/__init__.py`` imports all method modules eagerly,
so by the time anything calls ``get(category, name)`` the registry is
populated.

Categories are strings (not an enum) so the registry stays open to new method
families without code changes here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from biodisc.multiverse._types import StageContext

# Method callables share a uniform signature: take a StageContext + kwargs, return
# a category-specific result. Pre-Phase 8 categories (imputation / drift /
# normalisation / transformation / scaling) return ``pd.DataFrame``; multivariate
# methods return ``MultivariateResult``. Callers know the type per category.
MethodCallable = Callable[..., Any]


@dataclass(frozen=True)
class MethodSpec:
    """One row in the method registry."""

    name: str  # short identifier, e.g. "halfmin"
    callable: MethodCallable
    description: str  # one-line human-readable summary
    citation: str  # primary publication, e.g. "Lazar 2016 J Proteome Res"
    category: str  # "imputation" | "drift" | "normalisation" | "transformation" | "multivariate"


# Internal registry — keyed on f"{category}.{name}" so we can lookup by either.
_METHODS: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> None:
    """Add a method to the registry. Idempotent — re-registering with the same
    name + category overwrites the existing entry (useful for hot-reload in
    notebooks).
    """
    key = f"{spec.category}.{spec.name}"
    _METHODS[key] = spec


def get(category: str, name: str) -> MethodSpec:
    """Look up a method by category + name, raise if unknown."""
    key = f"{category}.{name}"
    if key not in _METHODS:
        available = sorted(s.name for s in _METHODS.values() if s.category == category)
        raise KeyError(
            f"Unknown {category} method: '{name}'. Available: {available}"
        )
    return _METHODS[key]


def list_methods(category: str | None = None) -> list[MethodSpec]:
    """Return registered methods, optionally filtered by category."""
    if category is None:
        return list(_METHODS.values())
    return [s for s in _METHODS.values() if s.category == category]


def apply(category: str, name: str, ctx: StageContext, **params: object) -> Any:
    """Convenience: look up + call. Most call-sites use this.

    Return type is category-specific: ``pd.DataFrame`` for matrix-transforming
    methods (imputation, drift, normalisation, transformation, scaling),
    ``MultivariateResult`` for multivariate-discovery methods.
    """
    spec = get(category, name)
    return spec.callable(ctx, **params)
