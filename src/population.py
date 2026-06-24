"""Population creation, elitism, and diversity metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.individual import Individual, create_individual


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def create_initial_population(
    n_cities: int,
    pop_size: int,
    *,
    rng: np.random.Generator | None = None,
    generation: int = 0,
) -> list[Individual]:
    """Create a founder population of random TSP permutations."""
    if n_cities < 2:
        raise ValueError("n_cities must be at least 2")
    if pop_size < 1:
        raise ValueError("pop_size must be at least 1")

    rng = _rng(rng)
    return [
        create_individual(rng.permutation(n_cities), generation=generation)
        for _ in range(pop_size)
    ]


def create_bitstring_population(
    n_bits: int,
    pop_size: int,
    *,
    rng: np.random.Generator | None = None,
    generation: int = 0,
) -> list[Individual]:
    """Create a founder population of random binary chromosomes."""
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1")
    if pop_size < 1:
        raise ValueError("pop_size must be at least 1")

    rng = _rng(rng)
    return [
        create_individual(rng.integers(0, 2, size=n_bits, dtype=int), generation=generation)
        for _ in range(pop_size)
    ]


def elitism(
    population: Sequence[Individual],
    n_elite: int,
    *,
    maximize: bool = False,
) -> list[Individual]:
    """Return the n best individuals for the configured objective direction."""
    if n_elite < 0:
        raise ValueError("n_elite must not be negative")
    if n_elite == 0:
        return []
    return sorted(
        population,
        key=lambda individual: individual.fitness,
        reverse=maximize,
    )[:n_elite]


def _undirected_edges(genes: np.ndarray) -> set[tuple[int, int]]:
    route = np.asarray(genes, dtype=int)
    return {
        tuple(sorted((int(route[i]), int(route[(i + 1) % len(route)]))))
        for i in range(len(route))
    }


def calculate_diversity(population: Sequence[Individual]) -> float:
    """Calculate average pairwise edge-difference diversity in [0, 1]."""
    if len(population) < 2:
        return 0.0

    edge_sets = [_undirected_edges(individual.genes) for individual in population]
    n_edges = len(edge_sets[0])
    if n_edges == 0:
        return 0.0

    edge_index = {
        edge: idx
        for idx, edge in enumerate(sorted(set().union(*edge_sets)))
    }
    edge_matrix = np.zeros((len(edge_sets), len(edge_index)), dtype=np.int16)
    for row, edges in enumerate(edge_sets):
        for edge in edges:
            edge_matrix[row, edge_index[edge]] = 1

    shared_counts = edge_matrix @ edge_matrix.T
    upper_triangle = np.triu_indices(len(edge_sets), k=1)
    distances = 1.0 - (shared_counts[upper_triangle] / n_edges)
    return float(np.mean(distances))


def calculate_bitstring_diversity(population: Sequence[Individual]) -> float:
    """Calculate average normalized Hamming distance in [0, 1]."""
    if len(population) < 2:
        return 0.0

    genes = np.asarray([individual.genes for individual in population], dtype=int)
    if genes.ndim != 2 or genes.shape[1] == 0:
        return 0.0
    if np.any((genes != 0) & (genes != 1)):
        raise ValueError("bitstring diversity requires only 0/1 genes")

    distances: list[float] = []
    for i in range(len(genes)):
        comparisons = genes[i + 1 :]
        if len(comparisons) == 0:
            continue
        distances.extend(np.mean(comparisons != genes[i], axis=1).tolist())
    return float(np.mean(distances)) if distances else 0.0
