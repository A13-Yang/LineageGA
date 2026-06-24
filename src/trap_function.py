"""Deceptive concatenated trap functions for binary GAs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrapFunction:
    """Concatenated deceptive trap benchmark.

    A block reaches the global optimum at all ones. For every non-optimal
    number of ones, the local slope points toward all zeros, which creates the
    deceptive basin used in GA benchmark studies.
    """

    block_size: int = 5
    n_blocks: int = 5

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be at least 1")

    @property
    def n_bits(self) -> int:
        """Total chromosome length."""
        return self.block_size * self.n_blocks

    @property
    def optimal_fitness(self) -> float:
        """Known global optimum fitness."""
        return 1.0

    def block_fitness(self, ones: int) -> float:
        """Return normalized deceptive fitness for one trap block."""
        if not 0 <= ones <= self.block_size:
            raise ValueError("ones must be between 0 and block_size")
        if ones == self.block_size:
            return 1.0
        return float((self.block_size - 1 - ones) / self.block_size)

    def fitness(self, bitstring: Iterable[int] | np.ndarray) -> float:
        """Return average concatenated trap fitness in [0, 1]."""
        bits = _as_bits(bitstring, self.n_bits)
        blocks = bits.reshape(self.n_blocks, self.block_size)
        scores = [self.block_fitness(int(block.sum())) for block in blocks]
        return float(np.mean(scores))


def trap_fitness(
    bitstring: Iterable[int] | np.ndarray,
    *,
    block_size: int = 5,
) -> float:
    """Convenience wrapper for a full-length concatenated trap chromosome."""
    bits = np.asarray(bitstring, dtype=int)
    if bits.ndim != 1 or len(bits) % block_size != 0:
        raise ValueError("bitstring length must be divisible by block_size")
    return TrapFunction(block_size=block_size, n_blocks=len(bits) // block_size).fitness(bits)


def _as_bits(bitstring: Iterable[int] | np.ndarray, expected_length: int) -> np.ndarray:
    bits = np.asarray(list(bitstring) if not isinstance(bitstring, np.ndarray) else bitstring, dtype=int)
    if bits.ndim != 1 or len(bits) != expected_length:
        raise ValueError(f"bitstring must be one-dimensional with length {expected_length}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bitstring must contain only 0/1 values")
    return bits
