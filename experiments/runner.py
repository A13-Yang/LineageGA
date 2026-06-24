"""Phase 3 experiment runner and artifact writer."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from experiments.config import DEFAULT_RESULT_DIR, ExperimentConfig
from src.fitness import calculate_tour_length
from src.ga_engine import ExperimentResult, GAEngine, GenerationRecord
from src.lineage import LineageTracker
from src.tsplib_parser import parse_opt_tour, parse_tsp


GENERATION_FIELDS = [
    "experiment_code",
    "strategy",
    "instance",
    "problem_type",
    "problem_param_n",
    "problem_param_k",
    "problem_param_block_size",
    "problem_param_n_blocks",
    "seed",
    "run_id",
    "generation",
    "best_fitness",
    "avg_fitness",
    "worst_fitness",
    "diversity",
    "avg_lc",
    "lc_fitness_correlation",
    "upset_offspring_rate",
    "best_individual_id",
]

INDIVIDUAL_FIELDS = [
    "experiment_code",
    "strategy",
    "instance",
    "problem_type",
    "problem_param_n",
    "problem_param_k",
    "problem_param_block_size",
    "problem_param_n_blocks",
    "seed",
    "run_id",
    "generation",
    "id",
    "fitness",
    "lc",
    "ancestry_entropy",
    "effective_founders",
    "ancestry_size",
    "parent_ids",
]

SUMMARY_FIELDS = [
    "experiment_code",
    "strategy",
    "instance",
    "problem_type",
    "problem_param_n",
    "problem_param_k",
    "problem_param_block_size",
    "problem_param_n_blocks",
    "seed",
    "run_id",
    "status",
    "final_best_fitness",
    "optimal_length",
    "gap_to_optimal",
    "convergence_gen_5pct",
    "convergence_gen_10pct",
    "convergence_gen_15pct",
    "final_avg_fitness",
    "final_worst_fitness",
    "final_avg_lc",
    "final_diversity",
    "elapsed_time",
    "population_size",
    "n_generations",
    "crossover_type",
    "mutation_rate",
    "n_elites",
    "selection_strategy",
    "tournament_size",
    "elite_ratio",
    "snapshot_interval",
]


def run_single(
    config: ExperimentConfig,
    seed: int,
    *,
    write_artifacts: bool = True,
) -> ExperimentResult:
    """Run one GA experiment and optionally persist Phase 3 artifacts."""
    result = GAEngine(config.to_ga_config()).run(seed=seed)
    if write_artifacts:
        save_run_artifacts(config, result)
    return result


def run_batch(
    config: ExperimentConfig,
    n_runs: int,
    *,
    base_seed: int = 0,
    show_progress: bool = True,
    write_artifacts: bool = True,
    merge_outputs: bool = True,
) -> list[ExperimentResult]:
    """Run a sequence of seeds for one config."""
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    seeds = [base_seed + offset for offset in range(n_runs)]
    iterator: Iterable[int] = seeds
    if show_progress:
        iterator = tqdm(seeds, desc=f"{config.strategy}/{config.instance}", unit="run")

    results = [
        run_single(config, seed, write_artifacts=write_artifacts)
        for seed in iterator
    ]
    if write_artifacts and merge_outputs:
        merge_run_artifacts(config)
    return results


def run_id(config: ExperimentConfig, seed: int | None) -> str:
    """Return a stable run identifier."""
    return f"{config.strategy}_{config.instance}_seed{seed}"


def calculate_optimal_length(config: ExperimentConfig) -> float | None:
    """Return the known optimum objective value when available."""
    if config.problem_type.lower() == "trap":
        return 1.0
    if config.problem_type.lower() == "nk":
        value = config.problem_params.get("optimal_fitness")
        return None if value is None else float(value)

    opt_path = config.resolved_opt_tour_path
    if not opt_path.exists():
        return None

    instance = parse_tsp(config.resolved_tsp_path)
    optimal_tour = parse_opt_tour(opt_path)
    return float(calculate_tour_length(optimal_tour, instance.distance_matrix))


def calculate_convergence_generations(
    history: list[GenerationRecord],
    optimal_length: float | None,
    thresholds: Iterable[float],
    *,
    maximize: bool = False,
) -> dict[float, int | None]:
    """Calculate first generation within each optimality gap threshold."""
    if optimal_length is None:
        return {float(threshold): None for threshold in thresholds}

    convergence: dict[float, int | None] = {}
    for threshold in thresholds:
        target = (
            optimal_length * (1.0 - float(threshold))
            if maximize
            else optimal_length * (1.0 + float(threshold))
        )
        generation = next(
            (
                record.generation
                for record in history
                if _reaches_target(record.best_fitness, target, maximize=maximize)
            ),
            None,
        )
        convergence[float(threshold)] = generation
    return convergence


def _reaches_target(value: float, target: float, *, maximize: bool) -> bool:
    return value >= target if maximize else value <= target


def save_run_artifacts(config: ExperimentConfig, result: ExperimentResult) -> None:
    """Persist per-run CSV/JSON files. Safe for parallel writers."""
    output_dir = _run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(result.seed) if result.seed is not None else -1
    current_run_id = run_id(config, result.seed)

    _write_csv(
        output_dir / f"{current_run_id}_gen.csv",
        GENERATION_FIELDS,
        _generation_rows(config, result, current_run_id),
    )
    _write_csv(
        output_dir / f"{current_run_id}_ind.csv",
        INDIVIDUAL_FIELDS,
        _individual_rows(config, result, current_run_id),
    )

    summary = _summary_row(config, result, current_run_id)
    _write_csv(
        output_dir / f"{current_run_id}_summary.csv",
        SUMMARY_FIELDS,
        [summary],
    )
    (output_dir / f"{current_run_id}_summary.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "summary": summary,
                "founder_qualities": result.founder_qualities,
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def merge_run_artifacts(config: ExperimentConfig) -> None:
    """Merge per-run files into the Phase 3 aggregate CSV names."""
    result_dir = config.resolved_result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _run_dir(config)

    _merge_csv_files(
        sorted(source_dir.glob("*_gen.csv")),
        result_dir / f"{config.run_group}_gen.csv",
        GENERATION_FIELDS,
    )
    _merge_csv_files(
        sorted(source_dir.glob("*_ind.csv")),
        result_dir / f"{config.run_group}_ind.csv",
        INDIVIDUAL_FIELDS,
    )
    merge_summary(config.resolved_result_dir)


def merge_summary(result_dir: str | Path) -> None:
    """Merge all per-run summaries under result_dir into summary.csv."""
    result_dir = Path(result_dir)
    summary_files = sorted((result_dir / "runs").glob("*/*_summary.csv"))
    _merge_csv_files(summary_files, result_dir / "summary.csv", SUMMARY_FIELDS)


def validate_result_integrity(
    configs: Iterable[ExperimentConfig],
    *,
    expected_runs_per_config: int = 30,
    result_dir: str | Path | None = None,
) -> dict[str, object]:
    """Check Phase 3 output completeness after a batch finishes."""
    config_list = list(configs)
    resolved_result_dir = (
        Path(result_dir)
        if result_dir is not None
        else config_list[0].resolved_result_dir
        if config_list
        else DEFAULT_RESULT_DIR
    )
    summary_path = resolved_result_dir / "summary.csv"
    summary_rows = _read_csv_rows(summary_path)
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]

    groups: dict[str, dict[str, object]] = {}
    for config in config_list:
        group_rows = [
            row
            for row in ok_rows
            if row.get("strategy") == config.strategy
            and row.get("instance") == config.instance
        ]
        seeds = {row.get("seed", "") for row in group_rows}
        gen_path = resolved_result_dir / f"{config.run_group}_gen.csv"
        ind_path = resolved_result_dir / f"{config.run_group}_ind.csv"
        generation_counts = _generation_counts_by_seed(gen_path)
        expected_generations = config.n_generations + 1
        generation_counts_ok = all(
            count == expected_generations
            for count in generation_counts.values()
        )
        groups[config.run_group] = {
            "summary_rows": len(group_rows),
            "unique_seeds": len(seeds),
            "summary_rows_ok": len(group_rows) == expected_runs_per_config,
            "unique_seeds_ok": len(seeds) == expected_runs_per_config,
            "gen_file_exists": gen_path.exists(),
            "ind_file_exists": ind_path.exists(),
            "generation_counts_ok": generation_counts_ok,
            "generation_counts_by_seed": generation_counts,
        }

    expected_total = expected_runs_per_config * len(config_list)
    groups_ok = all(
        group["summary_rows_ok"]
        and group["unique_seeds_ok"]
        and group["gen_file_exists"]
        and group["ind_file_exists"]
        and group["generation_counts_ok"]
        for group in groups.values()
    )
    return {
        "summary_path": str(summary_path),
        "expected_total_runs": expected_total,
        "actual_ok_runs": len(ok_rows),
        "all_runs_present": len(ok_rows) == expected_total,
        "groups_ok": groups_ok,
        "groups": groups,
    }


def load_summary_frame(result_dir: str | Path = DEFAULT_RESULT_DIR):
    """Load summary.csv as a pandas DataFrame for exploratory analysis."""
    import pandas as pd

    return pd.read_csv(Path(result_dir) / "summary.csv")


def summarize_results(result_dir: str | Path = DEFAULT_RESULT_DIR):
    """Return a grouped pandas summary for quick Phase 3 inspection."""
    frame = load_summary_frame(result_dir)
    return (
        frame.groupby(["strategy", "instance"])
        .agg(
            runs=("seed", "count"),
            best_mean=("final_best_fitness", "mean"),
            best_std=("final_best_fitness", "std"),
            gap_mean=("gap_to_optimal", "mean"),
            gap_std=("gap_to_optimal", "std"),
            lc_mean=("final_avg_lc", "mean"),
            diversity_mean=("final_diversity", "mean"),
            elapsed_mean=("elapsed_time", "mean"),
        )
        .reset_index()
    )


def _run_dir(config: ExperimentConfig) -> Path:
    return config.resolved_result_dir / "runs" / config.run_group


def _generation_rows(
    config: ExperimentConfig,
    result: ExperimentResult,
    current_run_id: str,
) -> list[dict[str, object]]:
    metadata = _problem_metadata(config)
    return [
        {
            "experiment_code": config.experiment_code,
            "strategy": config.strategy,
            "instance": config.instance,
            **metadata,
            "seed": result.seed,
            "run_id": current_run_id,
            "generation": record.generation,
            "best_fitness": record.best_fitness,
            "avg_fitness": record.avg_fitness,
            "worst_fitness": record.worst_fitness,
            "diversity": record.diversity,
            "avg_lc": record.avg_lc,
            "lc_fitness_correlation": record.lc_fitness_correlation,
            "upset_offspring_rate": record.upset_offspring_rate,
            "best_individual_id": record.best_individual_id,
        }
        for record in result.history
    ]


def _individual_rows(
    config: ExperimentConfig,
    result: ExperimentResult,
    current_run_id: str,
) -> list[dict[str, object]]:
    tracker = LineageTracker(result.founder_qualities)
    metadata = _problem_metadata(config)
    rows: list[dict[str, object]] = []
    for record in result.history:
        for snapshot in record.individuals:
            rows.append(
                {
                    "experiment_code": config.experiment_code,
                    "strategy": config.strategy,
                    "instance": config.instance,
                    **metadata,
                    "seed": result.seed,
                    "run_id": current_run_id,
                    "generation": snapshot.generation,
                    "id": snapshot.id,
                    "fitness": snapshot.fitness,
                    "lc": snapshot.lc,
                    "ancestry_entropy": tracker.compute_ancestry_entropy(snapshot.ancestry),
                    "effective_founders": tracker.compute_effective_founders(snapshot.ancestry),
                    "ancestry_size": len(snapshot.ancestry),
                    "parent_ids": "|".join(str(parent_id) for parent_id in snapshot.parent_ids),
                }
            )
    return rows


def _summary_row(
    config: ExperimentConfig,
    result: ExperimentResult,
    current_run_id: str,
) -> dict[str, object]:
    optimal_length = calculate_optimal_length(config)
    convergence = calculate_convergence_generations(
        result.history,
        optimal_length,
        config.convergence_thresholds,
        maximize=config.is_maximization,
    )
    final_record = result.history[-1]
    gap = (
        None
        if optimal_length is None
        else _optimality_gap(
            final_record.best_fitness,
            optimal_length,
            maximize=config.is_maximization,
        )
    )
    metadata = _problem_metadata(config)
    return {
        "experiment_code": config.experiment_code,
        "strategy": config.strategy,
        "instance": config.instance,
        **metadata,
        "seed": result.seed,
        "run_id": current_run_id,
        "status": "ok",
        "final_best_fitness": final_record.best_fitness,
        "optimal_length": optimal_length,
        "gap_to_optimal": gap,
        "convergence_gen_5pct": convergence.get(0.05),
        "convergence_gen_10pct": convergence.get(0.10),
        "convergence_gen_15pct": convergence.get(0.15),
        "final_avg_fitness": final_record.avg_fitness,
        "final_worst_fitness": final_record.worst_fitness,
        "final_avg_lc": final_record.avg_lc,
        "final_diversity": final_record.diversity,
        "elapsed_time": result.elapsed_time,
        "population_size": config.population_size,
        "n_generations": config.n_generations,
        "crossover_type": config.crossover_type,
        "mutation_rate": config.mutation_rate,
        "n_elites": config.n_elites,
        "selection_strategy": config.selection_strategy,
        "tournament_size": config.tournament_size,
        "elite_ratio": config.elite_ratio,
        "snapshot_interval": config.snapshot_interval,
    }


def _optimality_gap(value: float, optimum: float, *, maximize: bool) -> float:
    if optimum == 0.0:
        return float("nan")
    if maximize:
        return float((optimum - value) / abs(optimum))
    return float((value - optimum) / abs(optimum))


def _problem_metadata(config: ExperimentConfig) -> dict[str, object]:
    params = config.problem_params
    return {
        "problem_type": config.problem_type,
        "problem_param_n": params.get("n"),
        "problem_param_k": params.get("k"),
        "problem_param_block_size": params.get("block_size"),
        "problem_param_n_blocks": params.get("n_blocks"),
    }


def _write_csv(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_stringify_row(row) for row in rows)


def _merge_csv_files(
    paths: list[Path],
    output_path: Path,
    fields: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for path in paths:
            with path.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _generation_counts_by_seed(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_csv_rows(path):
        seed = row.get("seed", "")
        counts[seed] = counts.get(seed, 0) + 1
    return counts


def _stringify_row(row: dict[str, object]) -> dict[str, object]:
    return {
        key: "" if value is None else value
        for key, value in row.items()
    }


def result_to_dict(result: ExperimentResult) -> dict[str, object]:
    """Return a compact in-memory representation useful for notebooks."""
    return {
        "seed": result.seed,
        "elapsed_time": result.elapsed_time,
        "final_best_fitness": result.final_best_individual.fitness,
        "history": [asdict(record) for record in result.history],
    }
