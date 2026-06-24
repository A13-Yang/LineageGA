"""Fitness helpers for Euclidean TSP tours."""

from __future__ import annotations

import numpy as np


def build_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Build a TSPLIB EUC_2D distance matrix from city coordinates."""
    coordinates = np.asarray(cities, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("cities must be a 2D array with shape (n_cities, 2)")

    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return np.floor(distances + 0.5).astype(np.float64)


def calculate_tour_length(genes: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Calculate the closed-tour length for a permutation of city indices."""
    route = np.asarray(genes, dtype=int)
    distances = np.asarray(dist_matrix, dtype=float)

    if route.ndim != 1:
        raise ValueError("genes must be a one-dimensional permutation")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("dist_matrix must be a square 2D matrix")
    if len(route) == 0:
        return 0.0
    if route.min() < 0 or route.max() >= distances.shape[0]:
        raise ValueError("genes contain an index outside dist_matrix")

    return float(distances[route, np.roll(route, -1)].sum())
