"""Dash app for exploring Phase 4 analysis outputs."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from analysis.statistics import Phase04Data, build_statistical_report, load_phase03_data
from analysis.visualize import (
    make_interactive_ancestry_evolution,
    make_interactive_convergence,
    make_interactive_lc_scatter,
    make_interactive_nk_k_effect,
    make_interactive_performance_heatmap,
    make_interactive_problem_lc_effect,
    make_interactive_run_trajectory_3d,
    make_interactive_upset_rate,
)
from experiments.config import DEFAULT_RESULT_DIR


APP_STYLE = {
    "fontFamily": "Microsoft JhengHei, Segoe UI, Arial, sans-serif",
    "background": "#F6F7F9",
    "color": "#18202A",
    "minHeight": "100vh",
    "padding": "18px",
}

PANEL_STYLE = {
    "background": "#FFFFFF",
    "border": "1px solid #D9DEE7",
    "borderRadius": "8px",
    "padding": "12px",
}

GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
    "gap": "12px",
    "alignItems": "end",
}


def create_app(result_dir: str | Path = DEFAULT_RESULT_DIR) -> Dash:
    """Create and configure the Dash app."""
    app = Dash(__name__, title="Phase 4 Analysis")
    app.layout = _layout(result_dir)
    _register_callbacks(app)
    return app


def _layout(result_dir: str | Path) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Phase 4 Analysis", style={"margin": "0", "fontSize": "24px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Result Dir"),
                                    dcc.Input(
                                        id="result-dir",
                                        type="text",
                                        value=str(result_dir),
                                        debounce=True,
                                        style={"width": "100%"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Instance"),
                                    dcc.Dropdown(id="instance-filter", clearable=True),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Strategy"),
                                    dcc.Dropdown(id="strategy-filter", multi=True),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Generation"),
                                    dcc.Slider(
                                        id="generation-filter",
                                        min=0,
                                        max=0,
                                        step=None,
                                        value=0,
                                        marks={0: "0"},
                                    ),
                                ]
                            ),
                            html.Button("Refresh", id="refresh", n_clicks=0),
                        ],
                        style=GRID_STYLE,
                    ),
                ],
                style=PANEL_STYLE,
            ),
            html.Div(id="status-line", style={"margin": "10px 2px", "fontSize": "13px"}),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Overview",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="trajectory-graph"),
                                    dcc.Graph(id="convergence-graph"),
                                    dcc.Graph(id="performance-heatmap"),
                                ],
                                style={"display": "grid", "gap": "12px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="LC/Fitness",
                        children=[dcc.Graph(id="lc-scatter")],
                    ),
                    dcc.Tab(
                        label="Lineage",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="ancestry-graph"),
                                    dcc.Graph(id="upset-graph"),
                                ],
                                style={"display": "grid", "gap": "12px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Extended",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="nk-k-effect"),
                                    dcc.Graph(id="problem-lc-effect"),
                                ],
                                style={"display": "grid", "gap": "12px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tables",
                        children=[
                            html.Div(
                                [
                                    html.Button("CSV", id="download-summary-button", n_clicks=0),
                                    dcc.Download(id="download-summary"),
                                ],
                                style={"margin": "10px 0"},
                            ),
                            dash_table.DataTable(
                                id="summary-table",
                                page_size=12,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "Consolas, Microsoft JhengHei, monospace",
                                    "fontSize": "12px",
                                    "padding": "6px",
                                    "textAlign": "left",
                                    "minWidth": "90px",
                                },
                                style_header={
                                    "background": "#EDF1F7",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                    ),
                ]
            ),
        ],
        style=APP_STYLE,
    )


def _register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("instance-filter", "options"),
        Output("instance-filter", "value"),
        Output("strategy-filter", "options"),
        Output("strategy-filter", "value"),
        Output("generation-filter", "min"),
        Output("generation-filter", "max"),
        Output("generation-filter", "value"),
        Output("generation-filter", "marks"),
        Output("status-line", "children"),
        Input("refresh", "n_clicks"),
        State("result-dir", "value"),
    )
    def refresh_controls(_n_clicks: int, result_dir: str):
        data = load_phase03_data(result_dir or DEFAULT_RESULT_DIR)
        frame = _widest_table(data)
        instances = _options(frame, "instance")
        strategies = _options(frame, "strategy")
        generations = _generation_values(data)
        generation_min = int(min(generations)) if generations else 0
        generation_max = int(max(generations)) if generations else 0
        generation_value = generation_max
        marks = _slider_marks(generations)
        status = _status_text(data)
        return (
            instances,
            instances[0]["value"] if instances else None,
            strategies,
            [option["value"] for option in strategies],
            generation_min,
            generation_max,
            generation_value,
            marks,
            status,
        )

    @app.callback(
        Output("trajectory-graph", "figure"),
        Output("convergence-graph", "figure"),
        Output("performance-heatmap", "figure"),
        Output("lc-scatter", "figure"),
        Output("ancestry-graph", "figure"),
        Output("upset-graph", "figure"),
        Output("nk-k-effect", "figure"),
        Output("problem-lc-effect", "figure"),
        Output("summary-table", "columns"),
        Output("summary-table", "data"),
        Input("result-dir", "value"),
        Input("refresh", "n_clicks"),
        Input("instance-filter", "value"),
        Input("strategy-filter", "value"),
        Input("generation-filter", "value"),
    )
    def update_dashboard(
        result_dir: str,
        _n_clicks: int,
        instance: str | None,
        strategies: list[str] | str | None,
        generation: int | None,
    ):
        data = load_phase03_data(result_dir or DEFAULT_RESULT_DIR)
        filtered = _filter_data(data, instance=instance, strategies=strategies)
        extended = _filter_data(data, instance=None, strategies=strategies)
        table = _summary_table(filtered.summary, result_dir or DEFAULT_RESULT_DIR)
        return (
            make_interactive_run_trajectory_3d(filtered.generations),
            make_interactive_convergence(filtered.generations),
            make_interactive_performance_heatmap(filtered.summary),
            make_interactive_lc_scatter(filtered, generation=generation),
            make_interactive_ancestry_evolution(filtered),
            make_interactive_upset_rate(filtered.generations),
            make_interactive_nk_k_effect(extended),
            make_interactive_problem_lc_effect(extended),
            _data_table_columns(table),
            table.to_dict("records"),
        )

    @app.callback(
        Output("download-summary", "data"),
        Input("download-summary-button", "n_clicks"),
        State("result-dir", "value"),
        State("instance-filter", "value"),
        State("strategy-filter", "value"),
        prevent_initial_call=True,
    )
    def download_summary(
        n_clicks: int,
        result_dir: str,
        instance: str | None,
        strategies: list[str] | str | None,
    ):
        if not n_clicks:
            return no_update
        data = load_phase03_data(result_dir or DEFAULT_RESULT_DIR)
        filtered = _filter_data(data, instance=instance, strategies=strategies)
        table = _summary_table(filtered.summary, result_dir or DEFAULT_RESULT_DIR)
        return dcc.send_data_frame(table.to_csv, "phase04_filtered_summary.csv", index=False)


def _filter_data(
    data: Phase04Data,
    *,
    instance: str | None,
    strategies: list[str] | str | None,
) -> Phase04Data:
    strategy_values = [strategies] if isinstance(strategies, str) else list(strategies or [])

    return Phase04Data(
        result_dir=data.result_dir,
        summary=_filter_frame(data.summary, instance, strategy_values),
        generations=_filter_frame(data.generations, instance, strategy_values),
        individuals=_filter_frame(data.individuals, instance, strategy_values),
    )


def _filter_frame(
    frame: pd.DataFrame,
    instance: str | None,
    strategies: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if instance and "instance" in result:
        result = result[result["instance"] == instance]
    if strategies and "strategy" in result:
        result = result[result["strategy"].isin(strategies)]
    return result


def _summary_table(summary: pd.DataFrame, result_dir: str | Path) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["metric", "value"])

    metric = "gap_to_optimal" if "gap_to_optimal" in summary and summary["gap_to_optimal"].notna().any() else "final_best_fitness"
    table = (
        summary.groupby(["instance", "strategy"], dropna=False)
        .agg(
            runs=("seed", "count"),
            best_mean=(metric, "mean"),
            best_std=(metric, "std"),
            final_avg_lc=("final_avg_lc", "mean"),
            final_diversity=("final_diversity", "mean"),
            elapsed_mean=("elapsed_time", "mean"),
        )
        .reset_index()
    )

    report = build_statistical_report(result_dir)
    h2 = report.h2_anova[["instance", "metric", "p_value", "effect_size"]].copy()
    if not h2.empty:
        h2 = h2.rename(
            columns={
                "metric": "anova_metric",
                "p_value": "anova_p_value",
                "effect_size": "anova_eta_squared",
            }
        )
        table = table.merge(h2, on="instance", how="left")
    return table.round(6)


def _widest_table(data: Phase04Data) -> pd.DataFrame:
    if not data.summary.empty:
        return data.summary
    if not data.generations.empty:
        return data.generations
    return data.individuals


def _options(frame: pd.DataFrame, column: str) -> list[dict[str, str]]:
    if frame.empty or column not in frame:
        return []
    values = sorted(str(value) for value in frame[column].dropna().unique())
    return [{"label": value, "value": value} for value in values]


def _generation_values(data: Phase04Data) -> list[int]:
    values: set[int] = set()
    for frame in (data.generations, data.individuals):
        if not frame.empty and "generation" in frame:
            values.update(int(value) for value in frame["generation"].dropna().unique())
    return sorted(values)


def _slider_marks(values: list[int]) -> dict[int, str]:
    if not values:
        return {0: "0"}
    if len(values) <= 8:
        return {int(value): str(int(value)) for value in values}
    selected = np.linspace(0, len(values) - 1, 8, dtype=int)
    return {int(values[index]): str(int(values[index])) for index in selected}


def _status_text(data: Phase04Data) -> str:
    return (
        f"summary={len(data.summary)} rows, "
        f"generations={len(data.generations)} rows, "
        f"individuals={len(data.individuals)} rows"
    )


def _data_table_columns(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": column, "id": column} for column in frame.columns]


app = create_app()
server = app.server


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    if hasattr(app, "run"):
        app.run(host="127.0.0.1", port=port, debug=False)
    else:
        app.run_server(host="127.0.0.1", port=port, debug=False)
