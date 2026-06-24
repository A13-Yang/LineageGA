"""Visualization helpers for Phase 4 reports and the Dash dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", ".tmp/matplotlib"))
try:
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
except OSError:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from analysis.statistics import (
    Phase04Data,
    load_phase03_data,
    nk_k_lc_fitness_correlations,
    problem_lc_effect_comparison,
)
from experiments.config import DEFAULT_RESULT_DIR


DEFAULT_FIGURE_DIR = Path("data/results/figures")
STRATEGY_COLORS = {
    "tournament": "#2D6CDF",
    "elite": "#17A398",
    "poor": "#F4A261",
    "random": "#7B2CBF",
}


def configure_plot_style() -> None:
    """Apply project-wide plotting defaults, including CJK-capable fonts."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft JhengHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def generate_all_figures(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate Phase 4 and Phase 5 static PNG figures."""
    configure_plot_style()
    data = load_phase03_data(result_dir)
    figure_dir = Path(output_dir) if output_dir is not None else Path(result_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_convergence_curve(data.generations, figure_dir / "fig01_convergence_curve.png"),
        plot_lc_evolution(data.generations, figure_dir / "fig02_lc_evolution.png"),
        plot_lc_vs_fitness(
            data.individuals,
            data.generations,
            figure_dir / "fig03_lc_vs_fitness.png",
        ),
        plot_diversity_curve(data.generations, figure_dir / "fig04_diversity_curve.png"),
        plot_upset_rate_curve(data.generations, figure_dir / "fig05_upset_rate_curve.png"),
        plot_final_performance_box(data.summary, figure_dir / "fig06_final_performance_box.png"),
        plot_convergence_thresholds(data.summary, figure_dir / "fig07_convergence_thresholds.png"),
        plot_lc_diversity_heatmap(data.generations, figure_dir / "fig08_lc_diversity_heatmap.png"),
        plot_nk_k_effect(data, figure_dir / "fig09_nk_k_effect.png"),
        plot_problem_lc_effect_comparison(data, figure_dir / "fig10_problem_lc_effect_comparison.png"),
    ]


def generate_interactive_figures(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Write the Phase 4 Plotly figures as standalone HTML files."""
    data = load_phase03_data(result_dir)
    figure_dir = Path(output_dir) if output_dir is not None else Path(result_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "interactive_01_trajectory_3d.html": make_interactive_run_trajectory_3d(data.generations),
        "interactive_02_lc_scatter.html": make_interactive_lc_scatter(data),
        "interactive_03_ancestry_evolution.html": make_interactive_ancestry_evolution(data),
        "interactive_04_performance_heatmap.html": make_interactive_performance_heatmap(data.summary),
        "interactive_05_nk_k_effect.html": make_interactive_nk_k_effect(data),
        "interactive_06_problem_lc_effect.html": make_interactive_problem_lc_effect(data),
    }
    paths: list[Path] = []
    for filename, figure in figures.items():
        path = figure_dir / filename
        figure.write_html(path, include_plotlyjs=True)
        paths.append(path)
    return paths


def plot_convergence_curve(generations: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 1: best-fitness convergence, mean +/- std across runs."""
    if generations.empty or "best_fitness" not in generations:
        return _placeholder_png(output_path, "Convergence curve", "No generation data")

    aggregated = _aggregate_generation_metric(generations, "best_fitness")
    return _line_with_band(
        aggregated,
        output_path,
        metric="best_fitness",
        ylabel="Best fitness",
        title="Convergence curve",
    )


def plot_lc_evolution(generations: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 2: average lineage concentration through generations."""
    if generations.empty or "avg_lc" not in generations:
        return _placeholder_png(output_path, "LC evolution", "No generation data")

    aggregated = _aggregate_generation_metric(generations, "avg_lc")
    return _line_with_band(
        aggregated,
        output_path,
        metric="avg_lc",
        ylabel="Average LC",
        title="Lineage concentration evolution",
    )


def plot_lc_vs_fitness(
    individuals: pd.DataFrame,
    generations: pd.DataFrame,
    output_path: str | Path,
    *,
    max_points: int = 6000,
) -> Path:
    """Figure 3: LC-fitness scatter with per-strategy trend lines."""
    if not individuals.empty and {"lc", "fitness"}.issubset(individuals):
        frame = individuals.rename(columns={"lc": "lc_value", "fitness": "fitness_value"}).copy()
        source_label = "individual snapshots"
    elif not generations.empty and {"avg_lc", "best_fitness"}.issubset(generations):
        frame = generations.rename(
            columns={"avg_lc": "lc_value", "best_fitness": "fitness_value"}
        ).copy()
        source_label = "generation averages"
    else:
        return _placeholder_png(output_path, "LC vs. fitness", "No LC/fitness data")

    frame = frame.dropna(subset=["lc_value", "fitness_value"])
    if frame.empty:
        return _placeholder_png(output_path, "LC vs. fitness", "No LC/fitness data")
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=0)

    instances = _ordered_values(frame, "instance")
    fig, axes = _instance_subplots(instances, width=6.0, height=4.2)
    for ax, instance in zip(axes, instances):
        subset = frame[frame["instance"] == instance] if "instance" in frame else frame
        for strategy, group in subset.groupby("strategy", dropna=False):
            color = _strategy_color(strategy)
            ax.scatter(
                group["lc_value"],
                group["fitness_value"],
                s=16,
                alpha=0.32,
                label=str(strategy),
                color=color,
                linewidths=0,
            )
            _plot_trend_line(ax, group["lc_value"], group["fitness_value"], color)

        annotation = _correlation_annotation(subset["lc_value"], subset["fitness_value"])
        ax.set_title(f"{instance} ({source_label})")
        ax.set_xlabel("LC")
        ax.set_ylabel("Fitness")
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(loc="best")

    return _save_matplotlib(fig, output_path)


def plot_diversity_curve(generations: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 4: edge-difference diversity through generations."""
    if generations.empty or "diversity" not in generations:
        return _placeholder_png(output_path, "Diversity curve", "No generation data")

    aggregated = _aggregate_generation_metric(generations, "diversity")
    return _line_with_band(
        aggregated,
        output_path,
        metric="diversity",
        ylabel="Edge-difference diversity",
        title="Diversity curve",
    )


def plot_upset_rate_curve(generations: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 5: offspring upset rate through generations."""
    if generations.empty or "upset_offspring_rate" not in generations:
        return _placeholder_png(output_path, "Upset offspring rate", "No generation data")

    aggregated = _aggregate_generation_metric(generations, "upset_offspring_rate")
    return _line_with_band(
        aggregated,
        output_path,
        metric="upset_offspring_rate",
        ylabel="Upset offspring rate",
        title="Upset offspring rate",
    )


def plot_final_performance_box(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 6: final performance distribution by strategy."""
    metric = _performance_metric(summary)
    if summary.empty or metric not in summary:
        return _placeholder_png(output_path, "Final performance", "No summary data")

    frame = summary.dropna(subset=[metric]).copy()
    if frame.empty:
        return _placeholder_png(output_path, "Final performance", "No summary data")

    instances = _ordered_values(frame, "instance")
    fig, axes = _instance_subplots(instances, width=5.6, height=4.0)
    for ax, instance in zip(axes, instances):
        subset = frame[frame["instance"] == instance] if "instance" in frame else frame
        palette = _palette_for(subset["strategy"])
        sns.boxplot(
            data=subset,
            x="strategy",
            y=metric,
            hue="strategy",
            palette=palette,
            ax=ax,
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=subset,
            x="strategy",
            y=metric,
            color="#1F2933",
            alpha=0.45,
            size=3,
            ax=ax,
        )
        ax.set_title(str(instance))
        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Final performance distribution", y=1.02)
    return _save_matplotlib(fig, output_path)


def plot_convergence_thresholds(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 7: generation needed to reach each optimality threshold."""
    threshold_columns = [
        column
        for column in ("convergence_gen_5pct", "convergence_gen_10pct", "convergence_gen_15pct")
        if column in summary
    ]
    if summary.empty or not threshold_columns:
        return _placeholder_png(output_path, "Convergence thresholds", "No summary data")

    frame = summary.melt(
        id_vars=[column for column in ("instance", "strategy") if column in summary],
        value_vars=threshold_columns,
        var_name="threshold",
        value_name="generation",
    ).dropna(subset=["generation"])
    if frame.empty:
        return _placeholder_png(output_path, "Convergence thresholds", "No threshold hits")

    frame["threshold"] = (
        frame["threshold"]
        .str.replace("convergence_gen_", "", regex=False)
        .str.replace("pct", "%", regex=False)
    )
    instances = _ordered_values(frame, "instance")
    fig, axes = _instance_subplots(instances, width=5.8, height=4.0)
    for ax, instance in zip(axes, instances):
        subset = frame[frame["instance"] == instance] if "instance" in frame else frame
        palette = _palette_for(subset["strategy"])
        try:
            sns.barplot(
                data=subset,
                x="threshold",
                y="generation",
                hue="strategy",
                palette=palette,
                errorbar="sd",
                ax=ax,
            )
        except (AttributeError, TypeError):
            sns.barplot(
                data=subset,
                x="threshold",
                y="generation",
                hue="strategy",
                palette=palette,
                ci="sd",
                ax=ax,
            )
        ax.set_title(str(instance))
        ax.set_xlabel("Optimality gap threshold")
        ax.set_ylabel("Generation")
        ax.legend(loc="best")

    fig.suptitle("Convergence threshold timing", y=1.02)
    return _save_matplotlib(fig, output_path)


def plot_lc_diversity_heatmap(generations: pd.DataFrame, output_path: str | Path) -> Path:
    """Figure 8: heatmap of mean LC by generation and strategy."""
    if generations.empty or not {"avg_lc", "generation", "strategy"}.issubset(generations):
        return _placeholder_png(output_path, "LC heatmap", "No generation data")

    instances = _ordered_values(generations, "instance")
    fig, axes = _instance_subplots(instances, width=6.0, height=4.4)
    for ax, instance in zip(axes, instances):
        subset = generations[generations["instance"] == instance] if "instance" in generations else generations
        pivot = subset.pivot_table(
            index="strategy",
            columns="generation",
            values="avg_lc",
            aggfunc="mean",
        )
        if pivot.empty:
            ax.axis("off")
            ax.set_title(str(instance))
            continue
        sns.heatmap(
            pivot,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar=True,
            ax=ax,
        )
        ax.set_title(str(instance))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Strategy")

    fig.suptitle("Mean LC heatmap", y=1.02)
    return _save_matplotlib(fig, output_path)


def plot_nk_k_effect(data: Phase04Data, output_path: str | Path) -> Path:
    """Figure 9: NK epistasis K versus LC-fitness correlation."""
    correlations = nk_k_lc_fitness_correlations(data)
    if correlations.empty:
        return _placeholder_png(output_path, "NK K effect", "No NK extension data")

    frame = correlations[correlations["method"] == "pearson"].dropna(
        subset=["problem_param_k", "coefficient"]
    )
    if frame.empty:
        return _placeholder_png(output_path, "NK K effect", "No NK correlation data")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for strategy, group in frame.groupby("strategy", dropna=False):
        group = group.sort_values("problem_param_k")
        ax.plot(
            group["problem_param_k"],
            group["coefficient"],
            marker="o",
            linewidth=2.0,
            color=_strategy_color(strategy),
            label=str(strategy),
        )
    ax.axhline(0.0, color="#8A94A6", linewidth=1.0, linestyle="--")
    ax.set_title("NK epistasis effect on LC-fitness correlation")
    ax.set_xlabel("K")
    ax.set_ylabel("Pearson r")
    ax.legend(loc="best")
    return _save_matplotlib(fig, output_path)


def plot_problem_lc_effect_comparison(data: Phase04Data, output_path: str | Path) -> Path:
    """Figure 10: compare LC-fitness effect by problem family."""
    comparison = problem_lc_effect_comparison(data)
    if comparison.empty:
        return _placeholder_png(
            output_path,
            "Problem LC effect comparison",
            "No comparable LC/fitness data",
        )

    frame = comparison[comparison["method"] == "pearson"].dropna(subset=["coefficient"])
    if frame.empty:
        return _placeholder_png(
            output_path,
            "Problem LC effect comparison",
            "No Pearson correlation data",
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = _palette_for(frame["strategy"])
    sns.barplot(
        data=frame,
        x="problem_type",
        y="coefficient",
        hue="strategy",
        palette=palette,
        ax=ax,
    )
    ax.axhline(0.0, color="#8A94A6", linewidth=1.0, linestyle="--")
    ax.set_title("LC-fitness effect across problem families")
    ax.set_xlabel("Problem")
    ax.set_ylabel("Pearson r")
    ax.legend(loc="best")
    return _save_matplotlib(fig, output_path)


def make_interactive_run_trajectory_3d(generations: pd.DataFrame) -> go.Figure:
    """Plotly figure: generation, best fitness, and average LC per run."""
    if generations.empty:
        return _empty_plotly_figure("Run trajectory")

    figure = go.Figure()
    group_columns = [column for column in ("instance", "strategy", "run_id") if column in generations]
    if group_columns:
        grouped = generations.sort_values("generation").groupby(group_columns, dropna=False)
    else:
        grouped = [((), generations.sort_values("generation"))]
    for keys, group in grouped:
        values = keys if isinstance(keys, tuple) else (keys,)
        labels = dict(zip(group_columns, values))
        strategy = labels.get("strategy", "run")
        name = " / ".join(str(labels.get(column, "")) for column in group_columns if labels.get(column, "") != "")
        figure.add_trace(
            go.Scatter3d(
                x=group["generation"],
                y=group["best_fitness"],
                z=group["avg_lc"],
                mode="lines",
                name=name,
                line={"color": _strategy_color(strategy), "width": 4},
                hovertemplate=(
                    "generation=%{x}<br>"
                    "best=%{y:.3f}<br>"
                    "avg_lc=%{z:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Run trajectory: generation, best fitness, average LC",
        scene={
            "xaxis_title": "Generation",
            "yaxis_title": "Best fitness",
            "zaxis_title": "Average LC",
        },
        legend_title_text="Run",
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
    )
    return figure


def make_interactive_convergence(generations: pd.DataFrame) -> go.Figure:
    """Plotly figure: per-run convergence lines."""
    if generations.empty:
        return _empty_plotly_figure("Convergence")
    return px.line(
        generations.sort_values("generation"),
        x="generation",
        y="best_fitness",
        color="strategy",
        facet_col="instance" if "instance" in generations else None,
        line_group="run_id" if "run_id" in generations else None,
        hover_data=[column for column in ("seed", "avg_lc", "diversity") if column in generations],
        color_discrete_map=STRATEGY_COLORS,
        title="Best-fitness convergence",
    )


def make_interactive_lc_scatter(
    data: Phase04Data,
    *,
    generation: int | None = None,
    max_points: int = 8000,
) -> go.Figure:
    """Plotly figure: LC vs fitness at a selected generation or all snapshots."""
    if not data.individuals.empty and {"lc", "fitness"}.issubset(data.individuals):
        frame = data.individuals.rename(columns={"lc": "LC", "fitness": "Fitness"}).copy()
    elif not data.generations.empty and {"avg_lc", "best_fitness"}.issubset(data.generations):
        frame = data.generations.rename(columns={"avg_lc": "LC", "best_fitness": "Fitness"}).copy()
    else:
        return _empty_plotly_figure("LC vs fitness")

    if generation is not None and "generation" in frame:
        frame = frame[frame["generation"] == generation]
    frame = frame.dropna(subset=["LC", "Fitness"])
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=0)
    if frame.empty:
        return _empty_plotly_figure("LC vs fitness")

    return px.scatter(
        frame,
        x="LC",
        y="Fitness",
        color="strategy",
        facet_col="instance" if "instance" in frame else None,
        hover_data=[column for column in ("generation", "seed", "run_id", "id") if column in frame],
        color_discrete_map=STRATEGY_COLORS,
        title="LC vs fitness",
    )


def make_interactive_ancestry_evolution(data: Phase04Data) -> go.Figure:
    """Plotly figure: average LC and diversity over time."""
    if data.generations.empty:
        return _empty_plotly_figure("Ancestry evolution")

    frame = data.generations.sort_values("generation")
    return px.line(
        frame,
        x="generation",
        y="avg_lc",
        color="strategy",
        facet_col="instance" if "instance" in frame else None,
        line_group="run_id" if "run_id" in frame else None,
        hover_data=[column for column in ("diversity", "upset_offspring_rate", "seed") if column in frame],
        color_discrete_map=STRATEGY_COLORS,
        title="Average LC evolution",
    )


def make_interactive_upset_rate(generations: pd.DataFrame) -> go.Figure:
    """Plotly figure: offspring upset rate lines."""
    if generations.empty:
        return _empty_plotly_figure("Upset offspring rate")
    return px.line(
        generations.sort_values("generation"),
        x="generation",
        y="upset_offspring_rate",
        color="strategy",
        facet_col="instance" if "instance" in generations else None,
        line_group="run_id" if "run_id" in generations else None,
        hover_data=[column for column in ("seed", "best_fitness", "avg_lc") if column in generations],
        color_discrete_map=STRATEGY_COLORS,
        title="Upset offspring rate",
    )


def make_interactive_performance_heatmap(summary: pd.DataFrame) -> go.Figure:
    """Plotly figure: strategy x instance mean performance heatmap."""
    metric = _performance_metric(summary)
    if summary.empty or metric not in summary:
        return _empty_plotly_figure("Performance heatmap")
    frame = (
        summary.dropna(subset=[metric])
        .groupby(["instance", "strategy"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    if frame.empty:
        return _empty_plotly_figure("Performance heatmap")
    pivot = frame.pivot(index="instance", columns="strategy", values=metric)
    return px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": metric.replace("_", " ").title()},
        title="Mean final performance",
    )


def make_interactive_nk_k_effect(data: Phase04Data) -> go.Figure:
    """Plotly figure: NK K value versus LC-fitness correlation."""
    correlations = nk_k_lc_fitness_correlations(data)
    frame = correlations[correlations["method"] == "pearson"].dropna(
        subset=["problem_param_k", "coefficient"]
    )
    if frame.empty:
        return _empty_plotly_figure("NK K effect")
    return px.line(
        frame.sort_values("problem_param_k"),
        x="problem_param_k",
        y="coefficient",
        color="strategy",
        markers=True,
        hover_data=[column for column in ("problem_param_n", "p_value", "n") if column in frame],
        color_discrete_map=STRATEGY_COLORS,
        title="NK K effect on LC-fitness correlation",
        labels={"problem_param_k": "K", "coefficient": "Pearson r"},
    )


def make_interactive_problem_lc_effect(data: Phase04Data) -> go.Figure:
    """Plotly figure: compare LC-fitness effect across problem families."""
    comparison = problem_lc_effect_comparison(data)
    frame = comparison[comparison["method"] == "pearson"].dropna(subset=["coefficient"])
    if frame.empty:
        return _empty_plotly_figure("Problem LC effect")
    return px.bar(
        frame,
        x="problem_type",
        y="coefficient",
        color="strategy",
        barmode="group",
        hover_data=[column for column in ("p_value", "n") if column in frame],
        color_discrete_map=STRATEGY_COLORS,
        title="LC-fitness effect across problem families",
        labels={"problem_type": "Problem", "coefficient": "Pearson r"},
    )


def _aggregate_generation_metric(generations: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        generations.dropna(subset=[metric])
        .groupby(["instance", "strategy", "generation"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": metric, "std": f"{metric}_std"})
    )


def _line_with_band(
    aggregated: pd.DataFrame,
    output_path: str | Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    instances = _ordered_values(aggregated, "instance")
    fig, axes = _instance_subplots(instances, width=6.0, height=4.0)
    for ax, instance in zip(axes, instances):
        subset = aggregated[aggregated["instance"] == instance] if "instance" in aggregated else aggregated
        for strategy, group in subset.groupby("strategy", dropna=False):
            group = group.sort_values("generation")
            x = group["generation"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            std = group[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            color = _strategy_color(strategy)
            ax.plot(x, y, label=str(strategy), color=color, linewidth=2.0)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0)
        ax.set_title(str(instance))
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig.suptitle(title, y=1.02)
    return _save_matplotlib(fig, output_path)


def _instance_subplots(
    instances: Iterable[object],
    *,
    width: float,
    height: float,
) -> tuple[plt.Figure, np.ndarray]:
    instance_list = list(instances) or ["all"]
    fig, axes = plt.subplots(
        1,
        len(instance_list),
        figsize=(width * len(instance_list), height),
        squeeze=False,
    )
    return fig, axes.ravel()


def _ordered_values(frame: pd.DataFrame, column: str) -> list[object]:
    if column not in frame:
        return ["all"]
    return sorted(frame[column].dropna().unique().tolist())


def _strategy_color(strategy: object) -> str:
    return STRATEGY_COLORS.get(str(strategy), "#475569")


def _palette_for(values: pd.Series) -> dict[str, str]:
    return {str(value): _strategy_color(value) for value in values.dropna().unique()}


def _plot_trend_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2:
        return
    coefficients = np.polyfit(clean["x"], clean["y"], deg=1)
    xs = np.linspace(clean["x"].min(), clean["x"].max(), 80)
    ys = coefficients[0] * xs + coefficients[1]
    ax.plot(xs, ys, color=color, linewidth=2.0)


def _correlation_annotation(x: pd.Series, y: pd.Series) -> str:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return "r=n/a"
    result = stats.pearsonr(clean["x"], clean["y"])
    return f"r={result.statistic:.3f}, p={result.pvalue:.3g}"


def _performance_metric(summary: pd.DataFrame) -> str:
    if not summary.empty and "gap_to_optimal" in summary and summary["gap_to_optimal"].notna().any():
        return "gap_to_optimal"
    return "final_best_fitness"


def _save_matplotlib(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _placeholder_png(output_path: str | Path, title: str, message: str) -> Path:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return _save_matplotlib(fig, output_path)


def _empty_plotly_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        annotations=[
            {
                "text": "No data",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return figure
