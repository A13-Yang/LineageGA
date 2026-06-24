"""Smoke run for the Phase 2 lineage-aware GA engine on eil51."""

from __future__ import annotations

from pathlib import Path

from src.fitness import calculate_tour_length
from src.ga_engine import GAConfig, GAEngine
from src.tsplib_parser import parse_opt_tour, parse_tsp


def main() -> None:
    data_dir = Path("data/tsplib")
    instance = parse_tsp(data_dir / "eil51.tsp")
    optimal_tour = parse_opt_tour(data_dir / "eil51.opt.tour")
    optimal_length = calculate_tour_length(optimal_tour, instance.distance_matrix)

    config = GAConfig(
        distance_matrix=instance.distance_matrix,
        population_size=120,
        n_generations=300,
        crossover_type="ox",
        mutation_rate=0.08,
        n_elites=2,
        selection_strategy="tournament",
        tournament_size=5,
        elite_ratio=0.2,
        ancestry_prune_threshold=0.001,
        snapshot_interval=None,
    )
    result = GAEngine(config).run(seed=42)

    initial = result.history[0]
    final = result.history[-1]
    gap = (final.best_fitness - optimal_length) / optimal_length

    print(f"instance={instance.name}")
    print(f"seed={result.seed}")
    print(f"optimal_length={optimal_length:.0f}")
    print(f"initial_best={initial.best_fitness:.0f}")
    print(f"final_best={final.best_fitness:.0f}")
    print(f"gap_to_optimal={gap:.2%}")
    print(f"initial_avg_lc={initial.avg_lc:.4f}")
    print(f"final_avg_lc={final.avg_lc:.4f}")
    print(f"elapsed_time={result.elapsed_time:.2f}s")
    print("generation,best,avg_lc,diversity,upset_rate")
    for record in result.history:
        if record.generation % 50 == 0:
            print(
                f"{record.generation},"
                f"{record.best_fitness:.0f},"
                f"{record.avg_lc:.4f},"
                f"{record.diversity:.4f},"
                f"{record.upset_offspring_rate:.4f}"
            )


if __name__ == "__main__":
    main()
