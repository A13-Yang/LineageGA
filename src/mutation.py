"""Mutation operators for permutation and binary chromosomes."""

from __future__ import annotations

import numpy as np

from src.individual import Individual


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def swap_mutation(
    genes: np.ndarray,
    p_mutation: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Swap two positions with probability p_mutation and return a new array."""
    if not 0.0 <= p_mutation <= 1.0:
        raise ValueError("p_mutation must be between 0 and 1")

    rng = _rng(rng)
    mutated = np.asarray(genes, dtype=int).copy()
    if len(mutated) < 2 or rng.random() >= p_mutation:
        return mutated

    i, j = rng.choice(len(mutated), size=2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    assert set(mutated.tolist()) == set(np.asarray(genes, dtype=int).tolist())
    return mutated


def mutate_individual(
    individual: Individual,
    p_mutation: float,
    *,
    rng: np.random.Generator | None = None,
) -> Individual:
    """Return a new Individual carrying a swap-mutated chromosome."""
    return individual.copy_with(genes=swap_mutation(individual.genes, p_mutation, rng=rng))


def bitflip_mutation(
    genes: np.ndarray,
    p_mutation: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Flip each bit independently with probability ``p_mutation``."""
    if not 0.0 <= p_mutation <= 1.0:
        raise ValueError("p_mutation must be between 0 and 1")

    rng = _rng(rng)
    mutated = np.asarray(genes, dtype=int).copy()
    if mutated.ndim != 1:
        raise ValueError("genes must be a one-dimensional bitstring")
    if np.any((mutated != 0) & (mutated != 1)):
        raise ValueError("bitstring genes must contain only 0/1 values")
    flip_mask = rng.random(len(mutated)) < p_mutation
    mutated[flip_mask] = 1 - mutated[flip_mask]
    return mutated
