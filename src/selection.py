"""Parent selection strategies."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np

from src.individual import Individual


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def tournament_selection(
    population: Sequence[Individual],
    *,
    k: int = 3,
    rng: np.random.Generator | None = None,
    maximize: bool = False,
) -> Individual:
    """Select one individual using tournament selection."""
    if not population:
        raise ValueError("population must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    rng = _rng(rng)
    sample_size = min(k, len(population))
    competitors = rng.choice(np.asarray(population, dtype=object), size=sample_size, replace=False)
    selector = max if maximize else min
    return selector(competitors.tolist(), key=lambda individual: individual.fitness)


def ranked_selection_pool(
    population: Sequence[Individual],
    *,
    top: bool,
    ratio: float = 0.2,
    maximize: bool = False,
) -> list[Individual]:
    """Return the top or bottom fitness-ranked breeding pool."""
    if len(population) < 2:
        raise ValueError("population must contain at least two individuals")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in the interval (0, 1]")

    pool_size = max(2, ceil(len(population) * ratio))
    pool_size = min(pool_size, len(population))
    ranked = sorted(
        population,
        key=lambda individual: individual.fitness,
        reverse=maximize,
    )
    return ranked[:pool_size] if top else ranked[-pool_size:]


def _select_two_without_replacement(
    population: Sequence[Individual],
    *,
    rng: np.random.Generator,
) -> tuple[Individual, Individual]:
    parents = rng.choice(np.asarray(population, dtype=object), size=2, replace=False)
    return parents[0], parents[1]


def select_parents(
    population: Sequence[Individual],
    strategy: str = "tournament",
    *,
    rng: np.random.Generator | None = None,
    **kwargs: object,
) -> tuple[Individual, Individual]:
    """Select two parents through a common strategy interface."""
    if len(population) < 2:
        raise ValueError("population must contain at least two individuals")

    rng = _rng(rng)
    normalized = strategy.lower()
    maximize = bool(kwargs.get("maximize", False))
    if normalized == "tournament":
        k = int(kwargs.get("k", 3))
        parent_a = tournament_selection(population, k=k, rng=rng, maximize=maximize)
        parent_b = tournament_selection(population, k=k, rng=rng, maximize=maximize)
        for _ in range(5):
            if parent_b.id != parent_a.id:
                break
            parent_b = tournament_selection(population, k=k, rng=rng, maximize=maximize)
        if parent_b.id == parent_a.id:
            alternatives = [individual for individual in population if individual.id != parent_a.id]
            parent_b = alternatives[int(rng.integers(len(alternatives)))]
        return parent_a, parent_b
    if normalized == "elite":
        ratio = float(kwargs.get("elite_ratio", kwargs.get("ratio", 0.2)))
        return _select_two_without_replacement(
            ranked_selection_pool(population, top=True, ratio=ratio, maximize=maximize),
            rng=rng,
        )
    if normalized == "poor":
        ratio = float(kwargs.get("elite_ratio", kwargs.get("ratio", 0.2)))
        return _select_two_without_replacement(
            ranked_selection_pool(population, top=False, ratio=ratio, maximize=maximize),
            rng=rng,
        )
    if normalized == "random":
        return _select_two_without_replacement(population, rng=rng)

    raise ValueError(f"unsupported selection strategy: {strategy}")
