"""NK landscape benchmark for binary genetic algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class NKLandscape:
    """Deterministic NK landscape with adjacent circular interactions.

    Each locus depends on itself and the next ``K`` loci on a circular
    chromosome. The per-locus lookup tables are seeded once, so the same
    ``n``, ``k``, and ``seed`` define a stable experimental instance.
    """

    n: int = 20
    k: int = 2
    seed: int | None = 0
    contributions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if not 0 <= self.k < self.n:
            raise ValueError("k must be in the interval [0, n)")

        table_width = 2 ** (self.k + 1)
        if self.contributions is None:
            tables = _rng(self.seed).random((self.n, table_width))
        else:
            tables = np.asarray(self.contributions, dtype=float)
            if tables.shape != (self.n, table_width):
                raise ValueError(
                    f"contributions must have shape {(self.n, table_width)}"
                )
            if np.any((tables < 0.0) | (tables > 1.0)):
                raise ValueError("contributions must be in [0, 1]")
        object.__setattr__(self, "contributions", tables.copy())

    def dependency_indices(self, locus: int) -> tuple[int, ...]:
        """Return the circular dependency indices for a locus."""
        if not 0 <= locus < self.n:
            raise ValueError("locus out of range")
        return tuple((locus + offset) % self.n for offset in range(self.k + 1))

    def fitness(self, bitstring: Iterable[int] | np.ndarray) -> float:
        """Return NK fitness in [0, 1] for a 0/1 bitstring."""
        bits = _as_bits(bitstring, self.n)
        total = 0.0
        for locus in range(self.n):
            table_index = 0
            for bit in bits[list(self.dependency_indices(locus))]:
                table_index = (table_index << 1) | int(bit)
            total += float(self.contributions[locus, table_index])
        return float(total / self.n)

    def brute_force_optimum(self, *, max_n: int = 22) -> tuple[np.ndarray, float]:
        """Return the exact optimum by enumeration for small landscapes."""
        if self.n > max_n:
            raise ValueError(f"brute force search is limited to n <= {max_n}")

        best_bits: np.ndarray | None = None
        best_fitness = -np.inf
        for candidate in product((0, 1), repeat=self.n):
            bits = np.fromiter(candidate, dtype=int, count=self.n)
            value = self.fitness(bits)
            if value > best_fitness:
                best_bits = bits
                best_fitness = value
        if best_bits is None:
            raise ArithmeticError("failed to enumerate NK landscape")
        return best_bits, float(best_fitness)


def random_bitstring(
    n: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a random 0/1 chromosome of length ``n``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    generator = np.random.default_rng() if rng is None else rng
    return generator.integers(0, 2, size=n, dtype=int)


def _as_bits(bitstring: Iterable[int] | np.ndarray, expected_length: int) -> np.ndarray:
    bits = np.asarray(list(bitstring) if not isinstance(bitstring, np.ndarray) else bitstring, dtype=int)
    if bits.ndim != 1 or len(bits) != expected_length:
        raise ValueError(f"bitstring must be one-dimensional with length {expected_length}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bitstring must contain only 0/1 values")
    return bits
