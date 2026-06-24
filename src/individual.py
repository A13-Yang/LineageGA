"""Individual representation and global ID generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Iterable

import numpy as np

_id_counter = count()


def next_individual_id() -> int:
    """Return the next globally unique individual ID."""
    return next(_id_counter)


def reset_individual_id_counter(start: int = 0) -> None:
    """Reset the ID counter, mainly for deterministic tests."""
    global _id_counter
    _id_counter = count(start)


@dataclass(eq=False)
class Individual:
    """A GA individual encoded as a TSP city permutation."""

    id: int
    genes: np.ndarray
    fitness: float = float("inf")
    generation: int = 0
    parent_ids: tuple[int, ...] = field(default_factory=tuple)
    ancestry: dict[int, float] = field(default_factory=dict)
    lc: float = 0.0

    def __post_init__(self) -> None:
        self.genes = np.asarray(self.genes, dtype=int).copy()
        self.parent_ids = tuple(self.parent_ids)
        self.ancestry = dict(self.ancestry)
        if not self.ancestry and not self.parent_ids:
            self.ancestry = {self.id: 1.0}

    def copy_with(
        self,
        *,
        genes: Iterable[int] | np.ndarray | None = None,
        fitness: float | None = None,
        generation: int | None = None,
        parent_ids: tuple[int, ...] | None = None,
        ancestry: dict[int, float] | None = None,
        lc: float | None = None,
        id: int | None = None,
    ) -> "Individual":
        """Create a new individual, preserving omitted fields."""
        return Individual(
            id=next_individual_id() if id is None else id,
            genes=self.genes if genes is None else np.asarray(genes, dtype=int),
            fitness=self.fitness if fitness is None else fitness,
            generation=self.generation if generation is None else generation,
            parent_ids=self.parent_ids if parent_ids is None else parent_ids,
            ancestry=self.ancestry if ancestry is None else ancestry,
            lc=self.lc if lc is None else lc,
        )


def create_individual(
    genes: Iterable[int] | np.ndarray,
    *,
    fitness: float = float("inf"),
    generation: int = 0,
    parent_ids: tuple[int, ...] = (),
    ancestry: dict[int, float] | None = None,
    lc: float = 0.0,
) -> Individual:
    """Create an Individual with an automatically assigned global ID."""
    return Individual(
        id=next_individual_id(),
        genes=np.asarray(genes, dtype=int),
        fitness=fitness,
        generation=generation,
        parent_ids=parent_ids,
        ancestry={} if ancestry is None else ancestry,
        lc=lc,
    )
