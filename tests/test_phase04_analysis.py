from pathlib import Path

import pandas as pd

from analysis.report import build_phase04_outputs
from analysis.statistics import build_statistical_report, load_phase03_data
from analysis.visualize import make_interactive_convergence
from dashboard.app import create_app


def _write_sample_phase03_results(result_dir: Path) -> None:
    strategies = ["tournament", "elite", "poor", "random"]
    base_gap = {
        "tournament": 0.08,
        "elite": 0.10,
        "poor": 0.22,
        "random": 0.18,
    }
    base_lc = {
        "tournament": 0.72,
        "elite": 0.82,
        "poor": 0.58,
        "random": 0.63,
    }
    summary_rows = []
    generation_rows = []
    individual_rows = []

    for strategy_index, strategy in enumerate(strategies):
        for run_index in range(3):
            seed = 1000 + run_index
            run_id = f"{strategy}_eil51_seed{seed}"
            gap = base_gap[strategy] + 0.01 * run_index
            final_best = 426.0 * (1.0 + gap)
            summary_rows.append(
                {
                    "experiment_code": chr(ord("A") + strategy_index),
                    "strategy": strategy,
                    "instance": "eil51",
                    "seed": seed,
                    "run_id": run_id,
                    "status": "ok",
                    "final_best_fitness": final_best,
                    "optimal_length": 426.0,
                    "gap_to_optimal": gap,
                    "convergence_gen_5pct": "",
                    "convergence_gen_10pct": 100 if gap <= 0.10 else "",
                    "convergence_gen_15pct": 50 if gap <= 0.15 else 100,
                    "final_avg_fitness": final_best + 30.0,
                    "final_worst_fitness": final_best + 80.0,
                    "final_avg_lc": base_lc[strategy] + 0.01 * run_index,
                    "final_diversity": 0.45 - base_lc[strategy] * 0.25 + run_index * 0.01,
                    "elapsed_time": 1.0 + run_index,
                    "population_size": 12,
                    "n_generations": 100,
                    "crossover_type": "ox",
                    "mutation_rate": 0.08,
                    "n_elites": 1,
                    "selection_strategy": strategy,
                    "tournament_size": 5,
                    "elite_ratio": 0.2,
                    "snapshot_interval": 50,
                }
            )
            for generation in (0, 50, 100):
                progress = generation / 100.0
                avg_lc = 0.5 + (base_lc[strategy] - 0.5) * progress + 0.01 * run_index
                best = final_best + (1.0 - progress) * 120.0
                generation_rows.append(
                    {
                        "experiment_code": chr(ord("A") + strategy_index),
                        "strategy": strategy,
                        "instance": "eil51",
                        "seed": seed,
                        "run_id": run_id,
                        "generation": generation,
                        "best_fitness": best,
                        "avg_fitness": best + 30.0,
                        "worst_fitness": best + 90.0,
                        "diversity": 0.6 - 0.25 * progress + 0.01 * run_index,
                        "avg_lc": avg_lc,
                        "lc_fitness_correlation": -0.4,
                        "upset_offspring_rate": 0.1 + 0.03 * progress,
                        "best_individual_id": run_index,
                    }
                )
                for individual_id in range(6):
                    lc = avg_lc + individual_id * 0.015
                    individual_rows.append(
                        {
                            "experiment_code": chr(ord("A") + strategy_index),
                            "strategy": strategy,
                            "instance": "eil51",
                            "seed": seed,
                            "run_id": run_id,
                            "generation": generation,
                            "id": individual_id,
                            "fitness": best + 45.0 - 40.0 * lc + individual_id,
                            "lc": lc,
                            "ancestry_entropy": 1.4 - lc * 0.25,
                            "effective_founders": 3.0 - lc * 0.8,
                            "ancestry_size": 4,
                            "parent_ids": "1|2",
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(result_dir / "summary.csv", index=False)
    pd.DataFrame(generation_rows).to_csv(result_dir / "sample_gen.csv", index=False)
    pd.DataFrame(individual_rows).to_csv(result_dir / "sample_ind.csv", index=False)


def test_phase04_statistics_builds_hypothesis_tables(tmp_path):
    _write_sample_phase03_results(tmp_path)

    report = build_statistical_report(tmp_path)

    assert report.notes == ()
    assert report.run_counts["runs"].sum() == 12
    assert {"pearson", "spearman"}.issubset(set(report.h1_correlations["method"]))
    assert not report.h2_anova.empty
    assert not report.h2_tukey.empty
    assert not report.h3_kruskal.empty
    assert "Phase 4 statistical report" in report.to_text()


def test_phase04_report_writes_static_figures_and_tables(tmp_path):
    _write_sample_phase03_results(tmp_path)

    outputs = build_phase04_outputs(
        tmp_path,
        output_dir=tmp_path / "figures",
        include_interactive=False,
    )

    assert len(outputs.static_figures) == 10
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.static_figures)
    assert all(path.exists() for path in outputs.table_paths)
    assert outputs.text_report_path.exists()


def test_phase04_interactive_and_dashboard_entrypoint(tmp_path):
    _write_sample_phase03_results(tmp_path)
    data = load_phase03_data(tmp_path)

    figure = make_interactive_convergence(data.generations)
    app = create_app(tmp_path)

    assert figure.data
    assert app.layout is not None
