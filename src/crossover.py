"""Crossover operators for permutation and binary chromosomes."""

from __future__ import annotations

import numpy as np


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _as_permutation(genes: np.ndarray) -> np.ndarray:
    route = np.asarray(genes, dtype=int)
    if route.ndim != 1:
        raise ValueError("genes must be a one-dimensional permutation")
    if len(np.unique(route)) != len(route):
        raise ValueError("genes must not contain duplicate cities")
    return route


def _cut_points(n_genes: int, rng: np.random.Generator) -> tuple[int, int]:
    if n_genes < 2:
        raise ValueError("crossover requires at least two genes")
    start, end = sorted(rng.choice(n_genes, size=2, replace=False))
    return int(start), int(end) + 1


def assert_valid_permutation(child: np.ndarray, reference: np.ndarray) -> None:
    """Assert that child contains exactly the same genes as reference."""
    assert len(child) == len(reference), "child length differs from parent"
    assert set(child.tolist()) == set(reference.tolist()), "child is not a legal permutation"


def order_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create one child using Order Crossover (OX)."""
    rng = _rng(rng)
    a = _as_permutation(parent_a)
    b = _as_permutation(parent_b)
    if set(a.tolist()) != set(b.tolist()):
        raise ValueError("parents must contain the same city set")

    start, end = _cut_points(len(a), rng)
    child = np.full(len(a), -1, dtype=int)
    child[start:end] = a[start:end]

    fixed = set(child[start:end].tolist())
    remaining = [gene for gene in b.tolist() if gene not in fixed]
    fill_positions = list(range(end, len(a))) + list(range(0, start))
    for position, gene in zip(fill_positions, remaining):
        child[position] = gene

    assert_valid_permutation(child, a)
    return child


def pmx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create one child using Partially Mapped Crossover (PMX)."""
    rng = _rng(rng)
    a = _as_permutation(parent_a)
    b = _as_permutation(parent_b)
    if set(a.tolist()) != set(b.tolist()):
        raise ValueError("parents must contain the same city set")

    start, end = _cut_points(len(a), rng)
    child = np.full(len(a), -1, dtype=int)
    child[start:end] = a[start:end]

    b_positions = {gene: idx for idx, gene in enumerate(b.tolist())}
    segment_values = set(child[start:end].tolist())

    for idx in range(start, end):
        gene = int(b[idx])
        if gene in segment_values:
            continue

        position = idx
        while True:
            mapped_gene = int(a[position])
            position = b_positions[mapped_gene]
            if child[position] == -1:
                child[position] = gene
                break

    for idx, gene in enumerate(b):
        if child[idx] == -1:
            child[idx] = int(gene)

    assert_valid_permutation(child, a)
    return child


def order_crossover_pair(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two OX children by swapping parent roles."""
    rng = _rng(rng)
    return (
        order_crossover(parent_a, parent_b, rng=rng),
        order_crossover(parent_b, parent_a, rng=rng),
    )


def pmx_crossover_pair(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two PMX children by swapping parent roles."""
    rng = _rng(rng)
    return (
        pmx_crossover(parent_a, parent_b, rng=rng),
        pmx_crossover(parent_b, parent_a, rng=rng),
    )


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    swap_probability: float = 0.5,
) -> np.ndarray:
    """Create one binary child by independently choosing each bit from a parent."""
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("swap_probability must be between 0 and 1")

    rng = _rng(rng)
    a = _as_bitstring(parent_a)
    b = _as_bitstring(parent_b)
    if len(a) != len(b):
        raise ValueError("parents must have the same chromosome length")

    mask = rng.random(len(a)) < swap_probability
    child = np.where(mask, a, b).astype(int)
    return child


def uniform_crossover_pair(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    swap_probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two complementary children using the same uniform mask."""
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("swap_probability must be between 0 and 1")

    rng = _rng(rng)
    a = _as_bitstring(parent_a)
    b = _as_bitstring(parent_b)
    if len(a) != len(b):
        raise ValueError("parents must have the same chromosome length")

    mask = rng.random(len(a)) < swap_probability
    return np.where(mask, a, b).astype(int), np.where(mask, b, a).astype(int)


def _as_bitstring(genes: np.ndarray) -> np.ndarray:
    bits = np.asarray(genes, dtype=int)
    if bits.ndim != 1:
        raise ValueError("genes must be a one-dimensional bitstring")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bitstring genes must contain only 0/1 values")
    return bits
