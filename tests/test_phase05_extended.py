import csv

import numpy as np
import pytest

from analysis.statistics import (
    load_phase03_data,
    nk_k_lc_fitness_correlations,
    problem_lc_effect_comparison,
)
from analysis.visualize import make_interactive_nk_k_effect
from experiments.config import config_a, predefined_phase05_configs
from experiments.runner import calculate_convergence_generations, merge_run_artifacts, run_single
from src.crossover import uniform_crossover_pair
from src.ga_engine import GAConfig, GAEngine, GenerationRecord, is_upset_offspring
from src.mutation import bitflip_mutation
from src.nk_landscape import NKLandscape
from src.trap_function import TrapFunction, trap_fitness


def test_nk_landscape_is_deterministic_and_bounded():
    landscape_a = NKLandscape(n=8, k=2, seed=7)
    landscape_b = NKLandscape(n=8, k=2, seed=7)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])

    assert landscape_a.fitness(bits) == pytest.approx(landscape_b.fitness(bits))
    assert 0.0 <= landscape_a.fitness(bits) <= 1.0
    assert landscape_a.dependency_indices(7) == (7, 0, 1)


def test_trap_function_rewards_global_optimum_but_deceives_locally():
    trap = TrapFunction(block_size=5, n_blocks=2)

    assert trap.fitness(np.ones(10, dtype=int)) == pytest.approx(1.0)
    assert trap.fitness(np.zeros(10, dtype=int)) == pytest.approx(0.8)
    assert trap_fitness(np.zeros(10, dtype=int), block_size=5) == pytest.approx(0.8)
    assert trap.fitness(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])) < 0.8


def test_uniform_crossover_and_bitflip_mutation_for_bitstrings():
    rng = np.random.default_rng(42)
    parent_a = np.zeros(12, dtype=int)
    parent_b = np.ones(12, dtype=int)

    child_a, child_b = uniform_crossover_pair(parent_a, parent_b, rng=rng)
    assert set(child_a.tolist()).issubset({0, 1})
    assert np.array_equal(child_a + child_b, np.ones(12, dtype=int))

    flipped = bitflip_mutation(parent_a, 1.0, rng=np.random.default_rng(1))
    assert np.array_equal(flipped, parent_b)


def test_ga_engine_runs_nk_as_maximization_problem():
    landscape = NKLandscape(n=12, k=2, seed=11)
    config = GAConfig(
        problem_type="nk",
        chromosome_length=12,
        fitness_function=landscape.fitness,
        maximize=True,
        population_size=24,
        n_generations=8,
        crossover_type="uniform",
        mutation_rate=1 / 12,
        n_elites=1,
        snapshot_interval=4,
    )

    result = GAEngine(config).run(seed=3)

    assert len(result.history) == 9
    assert 0.0 <= result.final_best_individual.fitness <= 1.0
    assert result.history[-1].best_fitness >= result.history[0].best_fitness
    assert all(0.0 <= individual.lc <= 1.0 for individual in result.final_population)


def test_maximization_upset_and_convergence_thresholds():
    parent_a = type("Parent", (), {"fitness": 0.5})()
    parent_b = type("Parent", (), {"fitness": 0.7})()
    assert is_upset_offspring(0.61, parent_a, parent_b, maximize=True)
    assert not is_upset_offspring(0.60, parent_a, parent_b, maximize=True)

    history = [
        GenerationRecord(0, 0.70, 0.60, 0.40, 0.5, 0.2, 0.0, 0.0, 1),
        GenerationRecord(1, 0.88, 0.70, 0.45, 0.4, 0.4, 0.1, 0.2, 2),
        GenerationRecord(2, 0.96, 0.80, 0.50, 0.3, 0.6, 0.2, 0.3, 3),
    ]
    convergence = calculate_convergence_generations(
        history,
        1.0,
        (0.05, 0.10, 0.15),
        maximize=True,
    )

    assert convergence[0.05] == 2
    assert convergence[0.10] == 2
    assert convergence[0.15] == 1


def test_phase05_experiment_config_and_runner_metadata(tmp_path):
    config = config_a(
        instance="trap_2",
        problem_type="trap",
        problem_params={"block_size": 4, "n_blocks": 2},
        result_dir=str(tmp_path),
        population_size=16,
        n_generations=4,
        crossover_type="uniform",
        mutation_rate=1 / 8,
        n_elites=1,
        snapshot_interval=2,
    )

    run_single(config, seed=700)
    merge_run_artifacts(config)

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["problem_type"] == "trap"
    assert rows[0]["problem_param_block_size"] == "4"
    assert rows[0]["problem_param_n_blocks"] == "2"
    assert rows[0]["optimal_length"] == "1.0"
    assert float(rows[0]["gap_to_optimal"]) >= 0.0


def test_phase05_config_matrix_and_analysis_tables(tmp_path):
    configs = predefined_phase05_configs(
        result_dir=str(tmp_path),
        population_size=8,
        n_generations=1,
        n_elites=1,
        snapshot_interval=1,
    )
    assert len(configs) == 28

    nk_config = config_a(
        instance="nk_N8_K2",
        problem_type="nk",
        problem_params={"n": 8, "k": 2, "seed": 31},
        result_dir=str(tmp_path),
        population_size=16,
        n_generations=3,
        crossover_type="uniform",
        mutation_rate=1 / 8,
        n_elites=1,
        snapshot_interval=1,
    )
    trap_config = config_a(
        instance="trap_2",
        problem_type="trap",
        problem_params={"block_size": 4, "n_blocks": 2},
        result_dir=str(tmp_path),
        population_size=16,
        n_generations=3,
        crossover_type="uniform",
        mutation_rate=1 / 8,
        n_elites=1,
        snapshot_interval=1,
    )
    run_single(nk_config, seed=801)
    run_single(trap_config, seed=802)
    merge_run_artifacts(nk_config)
    merge_run_artifacts(trap_config)

    data = load_phase03_data(tmp_path)
    nk_table = nk_k_lc_fitness_correlations(data)
    comparison = problem_lc_effect_comparison(data)
    figure = make_interactive_nk_k_effect(data)

    assert not nk_table.empty
    assert set(comparison["problem_type"]).issuperset({"nk", "trap"})
    assert figure.data
