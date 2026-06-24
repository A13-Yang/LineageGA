import csv

from experiments.config import ExperimentConfig, config_a, predefined_configs
from experiments.parallel import estimate_parallel_memory_mb, run_parallel
from experiments.runner import (
    calculate_convergence_generations,
    merge_run_artifacts,
    run_batch,
    run_single,
    validate_result_integrity,
)
from src.ga_engine import GenerationRecord


def _tiny_config(tmp_path, **overrides):
    values = {
        "result_dir": str(tmp_path),
        "population_size": 12,
        "n_generations": 3,
        "mutation_rate": 0.05,
        "n_elites": 1,
        "snapshot_interval": 2,
    }
    values.update(overrides)
    return config_a(**values)


def test_experiment_config_json_roundtrip(tmp_path):
    config = _tiny_config(tmp_path, instance="eil51")
    path = tmp_path / "config.json"

    config.to_json(path)
    loaded = ExperimentConfig.from_json(path)

    assert loaded == config
    assert loaded.run_group == "tournament_eil51"
    assert len(predefined_configs(instances=("eil51",), result_dir=str(tmp_path))) == 4


def test_run_single_writes_per_run_artifacts_and_merge_outputs(tmp_path):
    config = _tiny_config(tmp_path)

    result = run_single(config, seed=123)
    merge_run_artifacts(config)

    run_dir = tmp_path / "runs" / "tournament_eil51"
    assert (run_dir / "tournament_eil51_seed123_gen.csv").exists()
    assert (run_dir / "tournament_eil51_seed123_ind.csv").exists()
    assert (run_dir / "tournament_eil51_seed123_summary.json").exists()
    assert (tmp_path / "tournament_eil51_gen.csv").exists()
    assert (tmp_path / "tournament_eil51_ind.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert len(result.history) == 4

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["strategy"] == "tournament"
    assert rows[0]["instance"] == "eil51"
    assert rows[0]["status"] == "ok"


def test_run_batch_uses_consecutive_seeds(tmp_path):
    config = _tiny_config(tmp_path, n_generations=1)

    run_batch(config, n_runs=2, base_seed=500, show_progress=False)

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        seeds = {row["seed"] for row in csv.DictReader(file)}
    assert seeds == {"500", "501"}

    integrity = validate_result_integrity([config], expected_runs_per_config=2)
    assert integrity["all_runs_present"]
    assert integrity["groups_ok"]


def test_parallel_runner_isolates_runs_with_single_worker(tmp_path):
    config = _tiny_config(tmp_path, n_generations=1)

    statuses = run_parallel([config], n_runs=2, base_seed=900, n_jobs=1)

    assert [status.status for status in statuses] == ["ok", "ok"]
    assert (tmp_path / "summary.csv").exists()
    assert estimate_parallel_memory_mb(config, n_jobs=1) > 0.0


def test_convergence_generation_thresholds():
    history = [
        GenerationRecord(0, 130.0, 140.0, 150.0, 0.5, 0.2, 0.0, 0.0, 1),
        GenerationRecord(1, 112.0, 120.0, 130.0, 0.4, 0.4, 0.1, 0.2, 2),
        GenerationRecord(2, 104.0, 110.0, 120.0, 0.3, 0.6, 0.2, 0.3, 3),
    ]

    convergence = calculate_convergence_generations(history, 100.0, (0.05, 0.10, 0.15))

    assert convergence[0.05] == 2
    assert convergence[0.10] == 2
    assert convergence[0.15] == 1
