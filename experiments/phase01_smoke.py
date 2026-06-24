"""Phase 1 smoke run: a simple lineage-free GA on eil51."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.crossover import order_crossover, pmx_crossover
from src.fitness import calculate_tour_length
from src.mutation import swap_mutation
from src.population import create_initial_population, elitism
from src.selection import select_parents
from src.tsplib_parser import parse_tsp


@dataclass(frozen=True)
class SmokeResult:
    initial_best: float
    final_best: float
    best_history: list[float]


def run_phase01_smoke(
    *,
    seed: int = 42,
    population_size: int = 80,
    generations: int = 100,
    mutation_rate: float = 0.05,
    n_elites: int = 2,
    crossover: str = "ox",
) -> SmokeResult:
    """Run a compact GA loop to verify Phase 1 components work together."""
    rng = np.random.default_rng(seed)
    instance = parse_tsp("data/tsplib/eil51.tsp")
    population = create_initial_population(instance.dimension, population_size, rng=rng)

    for individual in population:
        individual.fitness = calculate_tour_length(individual.genes, instance.distance_matrix)

    best_history = [min(individual.fitness for individual in population)]
    crossover_fn = order_crossover if crossover.lower() == "ox" else pmx_crossover

    for generation in range(1, generations + 1):
        next_population = list(elitism(population, n_elites))

        while len(next_population) < population_size:
            parent_a, parent_b = select_parents(
                population,
                "tournament",
                k=3,
                rng=rng,
            )
            child_genes = crossover_fn(parent_a.genes, parent_b.genes, rng=rng)
            child_genes = swap_mutation(child_genes, mutation_rate, rng=rng)
            child = parent_a.copy_with(
                genes=child_genes,
                fitness=calculate_tour_length(child_genes, instance.distance_matrix),
                generation=generation,
                parent_ids=(parent_a.id, parent_b.id),
                ancestry={},
                lc=0.0,
            )
            next_population.append(child)

        population = next_population
        best_history.append(min(individual.fitness for individual in population))

    return SmokeResult(
        initial_best=best_history[0],
        final_best=best_history[-1],
        best_history=best_history,
    )


if __name__ == "__main__":
    result = run_phase01_smoke()
    print(f"initial_best={result.initial_best:.0f}")
    print(f"final_best={result.final_best:.0f}")
    print(f"improved={result.final_best < result.initial_best}")
