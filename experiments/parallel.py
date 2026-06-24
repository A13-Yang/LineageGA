"""Parallel execution helpers for Phase 3 experiment batches."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Iterable

from joblib import Parallel, delayed

from experiments.config import ExperimentConfig, predefined_configs, predefined_phase05_configs
from experiments.runner import merge_run_artifacts, run_single


@dataclass(frozen=True)
class RunStatus:
    """Outcome of one parallel run."""

    experiment_code: str
    strategy: str
    instance: str
    problem_type: str
    seed: int
    status: str
    elapsed_time: float | None = None
    final_best_fitness: float | None = None
    error: str | None = None
    traceback: str | None = None


def run_single_safe(config: ExperimentConfig, seed: int) -> RunStatus:
    """Run one seed and convert exceptions into data instead of crashing a batch."""
    try:
        result = run_single(config, seed, write_artifacts=True)
        return RunStatus(
            experiment_code=config.experiment_code,
            strategy=config.strategy,
            instance=config.instance,
            problem_type=config.problem_type,
            seed=seed,
            status="ok",
            elapsed_time=result.elapsed_time,
            final_best_fitness=result.final_best_individual.fitness,
        )
    except Exception as exc:  # pragma: no cover - exercised through failure injection.
        return RunStatus(
            experiment_code=config.experiment_code,
            strategy=config.strategy,
            instance=config.instance,
            problem_type=config.problem_type,
            seed=seed,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def run_parallel(
    configs: Iterable[ExperimentConfig],
    *,
    n_runs: int = 30,
    base_seed: int = 0,
    n_jobs: int = 10,
    prefer: str = "processes",
) -> list[RunStatus]:
    """Run a Phase 3 batch in parallel and merge successful output files."""
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")

    config_list = list(configs)
    jobs = [
        (config, base_seed + run_index)
        for config in config_list
        for run_index in range(n_runs)
    ]
    statuses = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(run_single_safe)(config, seed)
        for config, seed in jobs
    )

    for config in config_list:
        merge_run_artifacts(config)
    return statuses


def run_full_phase03(
    *,
    n_runs: int = 30,
    base_seed: int = 0,
    n_jobs: int = 10,
) -> list[RunStatus]:
    """Run the full 4 strategy x 2 instance Phase 3 matrix."""
    return run_parallel(
        predefined_configs(),
        n_runs=n_runs,
        base_seed=base_seed,
        n_jobs=n_jobs,
    )


def run_full_phase05(
    *,
    n_runs: int = 30,
    base_seed: int = 5000,
    n_jobs: int = 10,
) -> list[RunStatus]:
    """Run the complete Phase 5 NK + Trap extension matrix."""
    return run_parallel(
        predefined_phase05_configs(),
        n_runs=n_runs,
        base_seed=base_seed,
        n_jobs=n_jobs,
    )


def estimate_parallel_memory_mb(config: ExperimentConfig, *, n_jobs: int = 10) -> float:
    """Estimate memory pressure for n concurrent workers."""
    ga_config = config.to_ga_config()
    if ga_config.distance_matrix is not None:
        matrix_mb = ga_config.distance_matrix.nbytes / (1024 * 1024)
        chromosome_length = ga_config.distance_matrix.shape[0]
    else:
        matrix_mb = 0.0
        chromosome_length = ga_config.chromosome_length or 0
    population_mb = (
        config.population_size
        * chromosome_length
        * 8
        / (1024 * 1024)
    )
    per_worker_mb = matrix_mb + population_mb + 50.0
    return per_worker_mb * n_jobs
