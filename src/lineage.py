"""Lineage tracking for founder ancestry and concentration metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log2

from src.individual import Individual


class LineageTracker:
    """Compute AP dictionaries, founder quality scores, and LC values."""

    def __init__(
        self,
        founder_qualities: Mapping[int, float] | None = None,
        *,
        prune_threshold: float = 0.001,
    ) -> None:
        if prune_threshold < 0.0:
            raise ValueError("prune_threshold must not be negative")

        self.founder_qualities = dict(founder_qualities or {})
        self.prune_threshold = float(prune_threshold)

    def compute_founder_quality(
        self,
        population: Sequence[Individual],
        *,
        maximize: bool = False,
    ) -> dict[int, float]:
        """Compute founder quality q_f from objective ranks."""
        if not population:
            raise ValueError("population must not be empty")

        ranked = sorted(
            population,
            key=lambda individual: (
                -individual.fitness if maximize else individual.fitness,
                individual.id,
            ),
        )
        n_founders = len(ranked)
        if n_founders == 1:
            qualities = {ranked[0].id: 1.0}
        else:
            denominator = n_founders - 1
            qualities = {
                individual.id: 1.0 - (rank - 1) / denominator
                for rank, individual in enumerate(ranked, start=1)
            }

        self.founder_qualities = qualities
        return qualities

    def prune_ancestry(self, ancestry: Mapping[int, float]) -> dict[int, float]:
        """Drop tiny AP entries and renormalize the remaining proportions."""
        positive = {
            int(founder_id): float(proportion)
            for founder_id, proportion in ancestry.items()
            if proportion > 0.0
        }
        if not positive:
            raise ValueError("ancestry must contain at least one positive proportion")

        pruned = {
            founder_id: proportion
            for founder_id, proportion in positive.items()
            if proportion >= self.prune_threshold
        }
        if not pruned:
            founder_id, proportion = max(positive.items(), key=lambda item: item[1])
            pruned = {founder_id: proportion}

        total = sum(pruned.values())
        if total <= 0.0:
            raise ValueError("ancestry proportions must sum to a positive value")

        normalized = {
            founder_id: proportion / total
            for founder_id, proportion in pruned.items()
        }
        if abs(sum(normalized.values()) - 1.0) > 1e-9:
            raise ArithmeticError("normalized ancestry does not sum to 1.0")
        return normalized

    def compute_offspring_ancestry(
        self,
        parent_a: Individual,
        parent_b: Individual,
    ) -> dict[int, float]:
        """Merge parent AP dictionaries using a 50/50 inheritance assumption."""
        combined: dict[int, float] = {}
        for founder_id, proportion in parent_a.ancestry.items():
            combined[int(founder_id)] = combined.get(int(founder_id), 0.0) + 0.5 * float(proportion)
        for founder_id, proportion in parent_b.ancestry.items():
            combined[int(founder_id)] = combined.get(int(founder_id), 0.0) + 0.5 * float(proportion)

        return self.prune_ancestry(combined)

    def compute_lc(self, ancestry: Mapping[int, float]) -> float:
        """Compute lineage concentration LC_i = sum_f p_i,f * q_f."""
        missing = set(ancestry) - set(self.founder_qualities)
        if missing:
            raise KeyError(f"missing founder quality for IDs: {sorted(missing)}")
        return float(
            sum(
                float(proportion) * self.founder_qualities[int(founder_id)]
                for founder_id, proportion in ancestry.items()
            )
        )

    def compute_ancestry_entropy(self, ancestry: Mapping[int, float]) -> float:
        """Compute Shannon entropy of an AP dictionary using log base 2."""
        return float(
            -sum(
                float(proportion) * log2(float(proportion))
                for proportion in ancestry.values()
                if proportion > 0.0
            )
        )

    def compute_effective_founders(self, ancestry: Mapping[int, float]) -> float:
        """Compute effective founder count: 1 / sum_f p_i,f squared."""
        denominator = sum(float(proportion) ** 2 for proportion in ancestry.values())
        if denominator <= 0.0:
            return 0.0
        return float(1.0 / denominator)
