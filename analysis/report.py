"""Phase 4 report builder.

Run with:

    python -m analysis.report
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from analysis.statistics import (
    StatisticalReport,
    build_statistical_report,
    load_phase03_data,
    nk_k_lc_fitness_correlations,
    problem_lc_effect_comparison,
)
from analysis.visualize import generate_all_figures, generate_interactive_figures
from experiments.config import DEFAULT_RESULT_DIR


@dataclass(frozen=True)
class Phase04Output:
    """Paths created by the Phase 4 report command."""

    report: StatisticalReport
    output_dir: Path
    table_paths: list[Path]
    static_figures: list[Path]
    interactive_figures: list[Path]
    text_report_path: Path


def build_phase04_outputs(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    *,
    output_dir: str | Path | None = None,
    include_interactive: bool = True,
) -> Phase04Output:
    """Generate statistical tables, static PNGs, and optional Plotly HTML."""
    result_path = Path(result_dir)
    artifact_dir = Path(output_dir) if output_dir is not None else result_path / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report = build_statistical_report(result_path)
    table_paths = write_statistical_tables(report, artifact_dir)
    table_paths.extend(write_phase05_tables(result_path, artifact_dir))
    text_report_path = artifact_dir / "phase04_statistics.txt"
    text_report_path.write_text(report.to_text(), encoding="utf-8")

    static_figures = generate_all_figures(result_path, artifact_dir)
    interactive_figures = (
        generate_interactive_figures(result_path, artifact_dir)
        if include_interactive
        else []
    )

    return Phase04Output(
        report=report,
        output_dir=artifact_dir,
        table_paths=table_paths,
        static_figures=static_figures,
        interactive_figures=interactive_figures,
        text_report_path=text_report_path,
    )


def write_statistical_tables(report: StatisticalReport, output_dir: str | Path) -> list[Path]:
    """Write every report table as a CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in report.tables().items():
        path = output_path / f"statistics_{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_phase05_tables(result_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Write Phase 5 extension analysis tables as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = load_phase03_data(result_dir)
    tables = {
        "phase05_nk_k_lc_fitness.csv": nk_k_lc_fitness_correlations(data),
        "phase05_problem_lc_effect.csv": problem_lc_effect_comparison(data),
    }
    paths: list[Path] = []
    for filename, table in tables.items():
        path = output_path / filename
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 4 analysis outputs.")
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Directory containing Phase 3 CSV artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Phase 4 figures and analysis tables.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip Plotly HTML outputs.",
    )
    args = parser.parse_args(argv)

    outputs = build_phase04_outputs(
        args.result_dir,
        output_dir=args.output_dir,
        include_interactive=not args.no_interactive,
    )
    print(f"Phase 4 outputs written to: {outputs.output_dir}")
    print(f"Static PNG figures: {len(outputs.static_figures)}")
    print(f"Interactive HTML figures: {len(outputs.interactive_figures)}")
    print(f"Statistical tables: {len(outputs.table_paths)}")
    if outputs.report.notes:
        print("Notes:")
        for note in outputs.report.notes:
            print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
