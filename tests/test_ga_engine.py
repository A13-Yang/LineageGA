import numpy as np
import pytest

from src.ga_engine import GAConfig, GAEngine, is_upset_offspring
from src.individual import create_individual, reset_individual_id_counter
from src.tsplib_parser import parse_tsp


def _eil51_matrix():
    return parse_tsp("data/tsplib/eil51.tsp").distance_matrix


def _small_config(**overrides):
    values = {
        "distance_matrix": _eil51_matrix(),
        "population_size": 24,
        "n_generations": 12,
        "mutation_rate": 0.05,
        "n_elites": 1,
        "snapshot_interval": 1,
    }
    values.update(overrides)
    return GAConfig(**values)


def test_ga_engine_runs_50_generations_without_crashing():
    result = GAEngine(
        _small_config(population_size=30, n_generations=50, snapshot_interval=25)
    ).run(seed=7)

    assert len(result.history) == 51
    assert np.isfinite(result.final_best_individual.fitness)
    assert len(result.final_population) == 30
    assert result.history[-1].generation == 50


def test_elitism_preserves_previous_generation_best_id():
    result = GAEngine(_small_config(n_generations=8, n_elites=1)).run(seed=11)

    for previous, current in zip(result.history, result.history[1:]):
        current_ids = {snapshot.id for snapshot in current.individuals}
        assert previous.best_individual_id in current_ids


@pytest.mark.parametrize("strategy", ["elite", "tournament", "poor", "random"])
def test_all_breeding_strategies_run(strategy):
    result = GAEngine(
        _small_config(selection_strategy=strategy, n_generations=5)
    ).run(seed=13)

    assert len(result.final_population) == 24
    assert result.history[-1].best_fitness <= result.history[0].best_fitness


def test_lineage_data_exists_for_every_final_individual():
    result = GAEngine(_small_config(n_generations=10)).run(seed=17)

    for individual in result.final_population:
        assert individual.ancestry
        assert sum(individual.ancestry.values()) == pytest.approx(1.0, abs=1e-10)
        assert 0.0 <= individual.lc <= 1.0


def test_upset_offspring_detection_uses_minimization_parent_average():
    reset_individual_id_counter()
    parent_a = create_individual([0, 1, 2], fitness=10.0)
    parent_b = create_individual([2, 1, 0], fitness=14.0)

    assert is_upset_offspring(11.9, parent_a, parent_b)
    assert not is_upset_offspring(12.0, parent_a, parent_b)
    assert not is_upset_offspring(12.1, parent_a, parent_b)
