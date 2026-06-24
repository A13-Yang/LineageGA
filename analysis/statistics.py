"""Statistical analysis helpers for Phase 4 artifacts.

The functions in this module consume the CSV files written by Phase 3:

* ``summary.csv`` for run-level outcomes.
* ``*_gen.csv`` for generation-level traces.
* ``*_ind.csv`` for individual ancestry snapshots.

They intentionally stay close to those file schemas so analysis, reports, and
the Dash dashboard can share one loading path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from experiments.config import DEFAULT_RESULT_DIR


SUMMARY_COLUMNS = [
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

GENERATION_COLUMNS = [
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

INDIVIDUAL_COLUMNS = [
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

NUMERIC_COLUMNS = {
    "seed",
    "generation",
    "problem_param_n",
    "problem_param_k",
    "problem_param_block_size",
    "problem_param_n_blocks",
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
    "mutation_rate",
    "n_elites",
    "tournament_size",
    "elite_ratio",
    "snapshot_interval",
    "best_fitness",
    "avg_fitness",
    "worst_fitness",
    "diversity",
    "avg_lc",
    "lc_fitness_correlation",
    "upset_offspring_rate",
    "best_individual_id",
    "id",
    "fitness",
    "lc",
    "ancestry_entropy",
    "effective_founders",
    "ancestry_size",
}


@dataclass(frozen=True)
class Phase04Data:
    """Loaded Phase 3 result tables."""

    result_dir: Path
    summary: pd.DataFrame
    generations: pd.DataFrame
    individuals: pd.DataFrame

    @property
    def has_summary(self) -> bool:
        return not self.summary.empty

    @property
    def has_generations(self) -> bool:
        return not self.generations.empty

    @property
    def has_individuals(self) -> bool:
        return not self.individuals.empty


@dataclass(frozen=True)
class StatisticalReport:
    """Container for the Phase 4 hypothesis-test tables."""

    result_dir: Path
    run_counts: pd.DataFrame
    h1_correlations: pd.DataFrame
    h2_anova: pd.DataFrame
    h2_tukey: pd.DataFrame
    h3_kruskal: pd.DataFrame
    h3_correlations: pd.DataFrame
    notes: tuple[str, ...] = ()

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return report tables keyed by stable names."""
        return {
            "run_counts": self.run_counts,
            "h1_correlations": self.h1_correlations,
            "h2_anova": self.h2_anova,
            "h2_tukey": self.h2_tukey,
            "h3_kruskal": self.h3_kruskal,
            "h3_correlations": self.h3_correlations,
        }

    def to_text(self, *, max_rows: int = 12) -> str:
        """Render a compact plain-text report without optional dependencies."""
        sections = [f"Phase 4 statistical report: {self.result_dir}"]
        if self.notes:
            sections.append("Notes:\n" + "\n".join(f"- {note}" for note in self.notes))

        for name, table in self.tables().items():
            sections.append(_format_table(name, table, max_rows=max_rows))
        return "\n\n".join(sections)


def load_phase03_data(result_dir: str | Path = DEFAULT_RESULT_DIR) -> Phase04Data:
    """Load Phase 3 CSV artifacts from aggregate files or per-run fallbacks."""
    result_path = Path(result_dir)
    summary = _concat_csvs(
        _preferred_paths(result_path, "summary.csv", "*_summary.csv"),
        SUMMARY_COLUMNS,
    )
    generations = _concat_csvs(
        _preferred_paths(result_path, "*_gen.csv", "*_gen.csv"),
        GENERATION_COLUMNS,
    )
    individuals = _concat_csvs(
        _preferred_paths(result_path, "*_ind.csv", "*_ind.csv"),
        INDIVIDUAL_COLUMNS,
    )
    return Phase04Data(
        result_dir=result_path,
        summary=_normalize_problem_metadata(_coerce_numeric(summary)),
        generations=_normalize_problem_metadata(_coerce_numeric(generations)),
        individuals=_normalize_problem_metadata(_coerce_numeric(individuals)),
    )


def build_statistical_report(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
) -> StatisticalReport:
    """Build all Phase 4 statistical tables from Phase 3 artifacts."""
    data = load_phase03_data(result_dir)
    notes: list[str] = []
    if data.summary.empty:
        notes.append("summary.csv was not found or contained no rows.")
    if data.generations.empty:
        notes.append("generation CSV files were not found or contained no rows.")
    if data.individuals.empty:
        notes.append("individual snapshot CSV files were not found or contained no rows.")

    return StatisticalReport(
        result_dir=data.result_dir,
        run_counts=run_count_table(data.summary),
        h1_correlations=lc_fitness_correlations(data),
        h2_anova=strategy_anova(data.summary),
        h2_tukey=strategy_tukey_hsd(data.summary),
        h3_kruskal=ancestry_kruskal_tests(data),
        h3_correlations=lc_diversity_correlations(data.generations),
        notes=tuple(notes),
    )


def run_count_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Count successful runs by instance and strategy."""
    if summary.empty:
        return pd.DataFrame(columns=["instance", "strategy", "runs"])

    frame = summary.copy()
    if "status" in frame:
        frame = frame[frame["status"].fillna("ok") == "ok"]
    return (
        frame.groupby(["instance", "strategy"], dropna=False)
        .size()
        .reset_index(name="runs")
        .sort_values(["instance", "strategy"])
        .reset_index(drop=True)
    )


def lc_fitness_correlations(data: Phase04Data) -> pd.DataFrame:
    """Test H1: LC-fitness correlation with Pearson and Spearman tests."""
    if not data.individuals.empty and {"lc", "fitness"}.issubset(data.individuals):
        frame = data.individuals.rename(columns={"lc": "x", "fitness": "y"}).copy()
        source = "individual_snapshots"
    elif not data.generations.empty and {"avg_lc", "best_fitness"}.issubset(data.generations):
        frame = data.generations.rename(
            columns={"avg_lc": "x", "best_fitness": "y"}
        ).copy()
        source = "generation_averages"
    else:
        return _empty_correlation_frame()

    group_cols = [
        column
        for column in ("instance", "strategy", "generation")
        if column in frame.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in _iter_groups(frame, group_cols):
        clean = group[["x", "y"]].dropna()
        if not _has_correlation_support(clean["x"], clean["y"]):
            continue

        key_values = _group_values(group_cols, keys)
        rows.extend(
            _correlation_rows(
                clean["x"],
                clean["y"],
                hypothesis="H1",
                source=source,
                group_values=key_values,
            )
        )

    return _sort_or_empty(pd.DataFrame(rows), _correlation_columns())


def strategy_anova(summary: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Test H2 with one-way ANOVA across selection strategies per instance."""
    metric = metric or _choose_performance_metric(summary)
    if summary.empty or metric not in summary:
        return _empty_anova_frame()

    rows: list[dict[str, object]] = []
    for instance, group in summary.groupby("instance", dropna=False):
        groups = _metric_groups(group, metric)
        if len(groups) < 2:
            continue

        arrays = [values for _, values in groups]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.f_oneway(*arrays)

        rows.append(
            {
                "hypothesis": "H2",
                "instance": instance,
                "metric": metric,
                "method": "one_way_anova",
                "statistic": _statistic(result),
                "p_value": _pvalue(result),
                "effect_size": _eta_squared(groups),
                "effect_size_name": "eta_squared",
                "n": int(sum(len(values) for _, values in groups)),
                "groups": ",".join(name for name, _ in groups),
                "significant_0_05": _pvalue(result) < 0.05,
            }
        )

    return _sort_or_empty(pd.DataFrame(rows), _anova_columns())


def strategy_tukey_hsd(summary: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Run Tukey HSD post-hoc comparisons for H2."""
    metric = metric or _choose_performance_metric(summary)
    if summary.empty or metric not in summary:
        return _empty_tukey_frame()

    rows: list[dict[str, object]] = []
    for instance, group in summary.groupby("instance", dropna=False):
        groups = _metric_groups(group, metric)
        if len(groups) < 2:
            continue

        names = [name for name, _ in groups]
        arrays = [values for _, values in groups]
        try:
            result = stats.tukey_hsd(*arrays)
            for i, left in enumerate(names):
                for j in range(i + 1, len(names)):
                    right = names[j]
                    rows.append(
                        {
                            "hypothesis": "H2",
                            "instance": instance,
                            "metric": metric,
                            "method": "tukey_hsd",
                            "group_a": left,
                            "group_b": right,
                            "mean_a": float(np.mean(arrays[i])),
                            "mean_b": float(np.mean(arrays[j])),
                            "mean_diff": float(np.mean(arrays[i]) - np.mean(arrays[j])),
                            "statistic": float(result.statistic[i, j]),
                            "p_value": float(result.pvalue[i, j]),
                            "significant_0_05": float(result.pvalue[i, j]) < 0.05,
                        }
                    )
        except AttributeError:
            rows.extend(_pairwise_bonferroni_rows(instance, metric, names, arrays))

    return _sort_or_empty(pd.DataFrame(rows), _tukey_columns())


def ancestry_kruskal_tests(data: Phase04Data) -> pd.DataFrame:
    """Test H3 with non-parametric group comparisons."""
    rows: list[dict[str, object]] = []
    if not data.summary.empty and "final_diversity" in data.summary:
        rows.extend(
            _kruskal_rows(
                data.summary,
                metric="final_diversity",
                hypothesis="H3",
                source="summary",
            )
        )

    final_individuals = final_individual_summary(data.individuals)
    for metric in ("mean_effective_founders", "mean_ancestry_entropy", "mean_lc"):
        if metric in final_individuals:
            rows.extend(
                _kruskal_rows(
                    final_individuals,
                    metric=metric,
                    hypothesis="H3",
                    source="final_individual_snapshots",
                )
            )

    return _sort_or_empty(pd.DataFrame(rows), _kruskal_columns())


def lc_diversity_correlations(generations: pd.DataFrame) -> pd.DataFrame:
    """Measure the relationship between LC concentration and diversity."""
    if generations.empty or not {"avg_lc", "diversity"}.issubset(generations):
        return _empty_correlation_frame()

    rows: list[dict[str, object]] = []
    group_cols = [column for column in ("instance", "strategy") if column in generations]
    for keys, group in _iter_groups(generations, group_cols):
        clean = group[["avg_lc", "diversity"]].dropna()
        if not _has_correlation_support(clean["avg_lc"], clean["diversity"]):
            continue
        rows.extend(
            _correlation_rows(
                clean["avg_lc"],
                clean["diversity"],
                hypothesis="H3",
                source="generation_averages",
                group_values=_group_values(group_cols, keys),
                methods=("spearman",),
            )
        )

    return _sort_or_empty(pd.DataFrame(rows), _correlation_columns())


def nk_k_lc_fitness_correlations(data: Phase04Data) -> pd.DataFrame:
    """Summarize Phase 5 NK LC-fitness correlation by K and strategy."""
    frame, source = _lc_fitness_frame(data)
    if frame.empty:
        return _empty_phase05_correlation_frame()

    nk = frame[frame["problem_type"] == "nk"].dropna(
        subset=["problem_param_k", "lc_value", "fitness_value"]
    )
    if nk.empty:
        return _empty_phase05_correlation_frame()

    rows: list[dict[str, object]] = []
    group_cols = ["problem_param_n", "problem_param_k", "strategy"]
    for keys, group in _iter_groups(nk, group_cols):
        clean = group[["lc_value", "fitness_value"]].dropna()
        if not _has_correlation_support(clean["lc_value"], clean["fitness_value"]):
            continue
        group_values = _group_values(group_cols, keys)
        for row in _correlation_rows(
            clean["lc_value"],
            clean["fitness_value"],
            hypothesis="P5-NK",
            source=source,
            group_values={
                "instance": "nk",
                **group_values,
            },
        ):
            rows.append(row)

    columns = _phase05_correlation_columns()
    return _sort_or_empty(pd.DataFrame(rows), columns)


def problem_lc_effect_comparison(data: Phase04Data) -> pd.DataFrame:
    """Compare LC-fitness correlation across TSP, NK, and Trap problems."""
    frame, source = _lc_fitness_frame(data)
    if frame.empty:
        return _empty_problem_effect_frame()

    rows: list[dict[str, object]] = []
    for keys, group in _iter_groups(frame, ["problem_type", "strategy"]):
        clean = group[["lc_value", "fitness_value"]].dropna()
        if not _has_correlation_support(clean["lc_value"], clean["fitness_value"]):
            continue
        rows.extend(
            _correlation_rows(
                clean["lc_value"],
                clean["fitness_value"],
                hypothesis="P5-Compare",
                source=source,
                group_values=_group_values(["problem_type", "strategy"], keys),
            )
        )

    columns = _problem_effect_columns()
    return _sort_or_empty(pd.DataFrame(rows), columns)


def final_individual_summary(individuals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final-generation ancestry metrics to one row per run."""
    required = {"run_id", "generation", "strategy", "instance"}
    if individuals.empty or not required.issubset(individuals):
        return pd.DataFrame(
            columns=[
                "run_id",
                "strategy",
                "instance",
                "seed",
                "mean_effective_founders",
                "mean_ancestry_entropy",
                "mean_lc",
            ]
        )

    frame = individuals.copy()
    max_generation = frame.groupby("run_id")["generation"].transform("max")
    frame = frame[frame["generation"] == max_generation]
    aggregations = {
        "effective_founders": "mean",
        "ancestry_entropy": "mean",
        "lc": "mean",
    }
    available = {key: value for key, value in aggregations.items() if key in frame}
    if not available:
        return pd.DataFrame()

    result = (
        frame.groupby(["run_id", "strategy", "instance", "seed"], dropna=False)
        .agg(available)
        .reset_index()
        .rename(
            columns={
                "effective_founders": "mean_effective_founders",
                "ancestry_entropy": "mean_ancestry_entropy",
                "lc": "mean_lc",
            }
        )
    )
    return result


def _preferred_paths(result_path: Path, aggregate_glob: str, run_glob: str) -> list[Path]:
    if aggregate_glob == "summary.csv":
        aggregate = [result_path / "summary.csv"] if (result_path / "summary.csv").exists() else []
    else:
        aggregate = sorted(result_path.glob(aggregate_glob))
    if aggregate:
        return aggregate
    return sorted((result_path / "runs").glob(f"*/*{run_glob.replace('*', '')}"))


def _concat_csvs(paths: Iterable[Path], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _normalize_problem_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "problem_type" not in frame:
        frame["problem_type"] = np.nan
    if "instance" in frame:
        inferred = frame["instance"].map(_infer_problem_type)
        frame["problem_type"] = frame["problem_type"].fillna(inferred)
        if "problem_param_k" in frame:
            extracted_k = frame["instance"].astype(str).str.extract(r"_K(\d+)", expand=False)
            frame["problem_param_k"] = frame["problem_param_k"].fillna(
                pd.to_numeric(extracted_k, errors="coerce")
            )
        if "problem_param_n" in frame:
            extracted_n = frame["instance"].astype(str).str.extract(r"_N(\d+)", expand=False)
            frame["problem_param_n"] = frame["problem_param_n"].fillna(
                pd.to_numeric(extracted_n, errors="coerce")
            )
    frame["problem_type"] = frame["problem_type"].fillna("tsp")
    return frame


def _infer_problem_type(instance: object) -> str:
    text = str(instance).lower()
    if text.startswith("nk"):
        return "nk"
    if text.startswith("trap"):
        return "trap"
    return "tsp"


def _lc_fitness_frame(data: Phase04Data) -> tuple[pd.DataFrame, str]:
    if not data.individuals.empty and {"lc", "fitness"}.issubset(data.individuals):
        frame = data.individuals.rename(
            columns={"lc": "lc_value", "fitness": "fitness_value"}
        ).copy()
        return frame, "individual_snapshots"
    if not data.generations.empty and {"avg_lc", "best_fitness"}.issubset(data.generations):
        frame = data.generations.rename(
            columns={"avg_lc": "lc_value", "best_fitness": "fitness_value"}
        ).copy()
        return frame, "generation_averages"
    return pd.DataFrame(), "none"


def _choose_performance_metric(summary: pd.DataFrame) -> str:
    if not summary.empty and "gap_to_optimal" in summary and summary["gap_to_optimal"].notna().any():
        return "gap_to_optimal"
    return "final_best_fitness"


def _metric_groups(frame: pd.DataFrame, metric: str) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    for strategy, group in frame.groupby("strategy", dropna=False):
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            groups.append((str(strategy), values))
    return groups


def _kruskal_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
    hypothesis: str,
    source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for instance, group in frame.groupby("instance", dropna=False):
        groups = _metric_groups(group, metric)
        if len(groups) < 2:
            continue
        result = stats.kruskal(*(values for _, values in groups))
        rows.append(
            {
                "hypothesis": hypothesis,
                "source": source,
                "instance": instance,
                "metric": metric,
                "method": "kruskal_wallis",
                "statistic": _statistic(result),
                "p_value": _pvalue(result),
                "effect_size": _epsilon_squared(result, groups),
                "effect_size_name": "epsilon_squared",
                "n": int(sum(len(values) for _, values in groups)),
                "groups": ",".join(name for name, _ in groups),
                "significant_0_05": _pvalue(result) < 0.05,
            }
        )
    return rows


def _correlation_rows(
    x: pd.Series,
    y: pd.Series,
    *,
    hypothesis: str,
    source: str,
    group_values: dict[str, object],
    methods: tuple[str, ...] = ("pearson", "spearman"),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_functions: dict[str, Callable[[pd.Series, pd.Series], object]] = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
    }
    for method in methods:
        result = method_functions[method](x, y)
        row = {
            "hypothesis": hypothesis,
            "source": source,
            "method": method,
            "coefficient": _statistic(result),
            "p_value": _pvalue(result),
            "n": int(len(x)),
            "significant_0_05": _pvalue(result) < 0.05,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
        row.update(group_values)
        if method == "pearson" and hasattr(result, "confidence_interval"):
            try:
                interval = result.confidence_interval(confidence_level=0.95)
                row["ci_low"] = float(interval.low)
                row["ci_high"] = float(interval.high)
            except Exception:
                pass
        rows.append(row)
    return rows


def _pairwise_bonferroni_rows(
    instance: object,
    metric: str,
    names: list[str],
    arrays: list[np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = max(1, len(names) * (len(names) - 1) // 2)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            result = stats.ttest_ind(arrays[i], arrays[j], equal_var=False)
            p_value = min(1.0, _pvalue(result) * comparisons)
            rows.append(
                {
                    "hypothesis": "H2",
                    "instance": instance,
                    "metric": metric,
                    "method": "pairwise_t_bonferroni",
                    "group_a": left,
                    "group_b": right,
                    "mean_a": float(np.mean(arrays[i])),
                    "mean_b": float(np.mean(arrays[j])),
                    "mean_diff": float(np.mean(arrays[i]) - np.mean(arrays[j])),
                    "statistic": _statistic(result),
                    "p_value": p_value,
                    "significant_0_05": p_value < 0.05,
                }
            )
    return rows


def _eta_squared(groups: list[tuple[str, np.ndarray]]) -> float:
    values = np.concatenate([group_values for _, group_values in groups])
    grand_mean = float(np.mean(values))
    ss_between = sum(
        len(group_values) * (float(np.mean(group_values)) - grand_mean) ** 2
        for _, group_values in groups
    )
    ss_total = float(np.sum((values - grand_mean) ** 2))
    return float(ss_between / ss_total) if ss_total > 0.0 else np.nan


def _epsilon_squared(
    result: object,
    groups: list[tuple[str, np.ndarray]],
) -> float:
    n = sum(len(values) for _, values in groups)
    k = len(groups)
    if n <= k:
        return np.nan
    h = _statistic(result)
    return float(max(0.0, (h - k + 1.0) / (n - k)))


def _has_correlation_support(x: pd.Series, y: pd.Series) -> bool:
    return len(x) >= 3 and x.nunique(dropna=True) >= 2 and y.nunique(dropna=True) >= 2


def _iter_groups(frame: pd.DataFrame, group_cols: list[str]):
    if not group_cols:
        yield (), frame
        return
    yield from frame.groupby(group_cols, dropna=False)


def _group_values(group_cols: list[str], keys: object) -> dict[str, object]:
    if not group_cols:
        return {}
    if len(group_cols) == 1:
        keys = (keys,)
    return dict(zip(group_cols, keys))


def _statistic(result: object) -> float:
    value = getattr(result, "statistic", result[0] if isinstance(result, tuple) else np.nan)
    return float(value)


def _pvalue(result: object) -> float:
    value = getattr(result, "pvalue", result[1] if isinstance(result, tuple) else np.nan)
    return float(value)


def _sort_or_empty(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    sort_columns = [
        column
        for column in (
            "problem_type",
            "problem_param_k",
            "instance",
            "strategy",
            "generation",
            "method",
        )
        if column in frame
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    return frame[columns].reset_index(drop=True)


def _format_table(name: str, table: pd.DataFrame, *, max_rows: int) -> str:
    title = name.replace("_", " ").title()
    if table.empty:
        return f"{title}:\n(empty)"
    displayed = table.head(max_rows)
    suffix = "" if len(table) <= max_rows else f"\n... {len(table) - max_rows} more rows"
    return f"{title}:\n{displayed.to_string(index=False)}{suffix}"


def _correlation_columns() -> list[str]:
    return [
        "hypothesis",
        "source",
        "instance",
        "strategy",
        "generation",
        "method",
        "coefficient",
        "p_value",
        "ci_low",
        "ci_high",
        "n",
        "significant_0_05",
    ]


def _anova_columns() -> list[str]:
    return [
        "hypothesis",
        "instance",
        "metric",
        "method",
        "statistic",
        "p_value",
        "effect_size",
        "effect_size_name",
        "n",
        "groups",
        "significant_0_05",
    ]


def _tukey_columns() -> list[str]:
    return [
        "hypothesis",
        "instance",
        "metric",
        "method",
        "group_a",
        "group_b",
        "mean_a",
        "mean_b",
        "mean_diff",
        "statistic",
        "p_value",
        "significant_0_05",
    ]


def _kruskal_columns() -> list[str]:
    return [
        "hypothesis",
        "source",
        "instance",
        "metric",
        "method",
        "statistic",
        "p_value",
        "effect_size",
        "effect_size_name",
        "n",
        "groups",
        "significant_0_05",
    ]


def _empty_correlation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_correlation_columns())


def _empty_anova_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_anova_columns())


def _empty_tukey_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_tukey_columns())


def _phase05_correlation_columns() -> list[str]:
    return [
        "hypothesis",
        "source",
        "instance",
        "problem_param_n",
        "problem_param_k",
        "strategy",
        "method",
        "coefficient",
        "p_value",
        "ci_low",
        "ci_high",
        "n",
        "significant_0_05",
    ]


def _problem_effect_columns() -> list[str]:
    return [
        "hypothesis",
        "source",
        "problem_type",
        "strategy",
        "method",
        "coefficient",
        "p_value",
        "ci_low",
        "ci_high",
        "n",
        "significant_0_05",
    ]


def _empty_phase05_correlation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_phase05_correlation_columns())


def _empty_problem_effect_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_problem_effect_columns())
