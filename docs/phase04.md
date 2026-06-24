# Phase 4 實作紀錄
日期：2026-06-24

## 前置閱讀

本階段實作前已閱讀：

- `docs/phase01.md`：確認 TSPLIB parser、fitness、crossover、mutation、population、selection 的基礎模組與測試方式。
- `docs/phase02.md`：確認 `GAEngine`、`GAConfig`、`GenerationRecord`、`ExperimentResult`、AP/LC 與 lineage metric 的資料來源。
- `docs/phase03.md`：確認 Phase 3 的實驗矩陣、per-run CSV/JSON、aggregate CSV 與 `data/results` 輸出 schema。

Phase 4 因此沒有修改 GA 核心，而是建立在 Phase 3 的 `summary.csv`、`*_gen.csv`、`*_ind.csv` artefacts 之上。

## 新增與修改檔案

| 檔案 | 內容 |
|---|---|
| `analysis/statistics.py` | Phase 3 結果讀取、H1/H2/H3 統計檢定、效果量、`StatisticalReport` dataclass |
| `analysis/visualize.py` | 8 張靜態 PNG 圖表產生器、Plotly 互動圖工廠、中文字型與 300 DPI 輸出設定 |
| `analysis/report.py` | `python -m analysis.report` 一鍵輸出統計表、文字摘要、PNG 與互動 HTML |
| `dashboard/__init__.py` | Dashboard package 初始化 |
| `dashboard/app.py` | Plotly Dash Web UI，提供篩選、圖表、統計表與 CSV 匯出 |
| `tests/test_phase04_analysis.py` | Phase 4 統計、圖表、報告與 Dash app smoke tests |
| `pyproject.toml` | 將 `dashboard` 加入 setuptools package 清單 |

## 統計模組

`analysis/statistics.py` 會優先讀取 aggregate 檔：

- `data/results/summary.csv`
- `data/results/*_gen.csv`
- `data/results/*_ind.csv`

若 aggregate 檔不存在，會 fallback 到：

- `data/results/runs/*/*_summary.csv`
- `data/results/runs/*/*_gen.csv`
- `data/results/runs/*/*_ind.csv`

已實作的主要 API：

- `load_phase03_data(result_dir)`：讀取並正規化 Phase 3 CSV。
- `build_statistical_report(result_dir)`：建立完整統計報告。
- `lc_fitness_correlations(data)`：H1，LC 與 fitness 的 Pearson / Spearman correlation，含 p-value 與 Pearson 95% CI。
- `strategy_anova(summary)`：H2，各 selection strategy 對 final performance 的 one-way ANOVA。
- `strategy_tukey_hsd(summary)`：H2 post-hoc Tukey HSD；若 SciPy 版本沒有 `tukey_hsd`，會 fallback 到 Bonferroni 修正的 pairwise t-test。
- `ancestry_kruskal_tests(data)`：H3，final diversity、effective founders、ancestry entropy、mean LC 的 Kruskal-Wallis test。
- `lc_diversity_correlations(generations)`：H3，avg LC 與 diversity 的 Spearman correlation。

效果量：

- H2 ANOVA 使用 eta-squared。
- H3 Kruskal-Wallis 使用 epsilon-squared。

## 圖表模組

`analysis/visualize.py` 產生 8 張 PNG，預設輸出到 `data/results/figures/`：

| 檔名 | 內容 |
|---|---|
| `fig01_convergence_curve.png` | best fitness 收斂曲線，跨 runs mean ± std |
| `fig02_lc_evolution.png` | avg LC 隨 generation 演化 |
| `fig03_lc_vs_fitness.png` | LC vs fitness scatter 與趨勢線 |
| `fig04_diversity_curve.png` | edge-difference diversity 曲線 |
| `fig05_upset_rate_curve.png` | upset offspring rate 曲線 |
| `fig06_final_performance_box.png` | final performance box/strip plot |
| `fig07_convergence_thresholds.png` | 5%/10%/15% convergence generation |
| `fig08_lc_diversity_heatmap.png` | strategy × generation 的 mean LC heatmap |

另有互動 HTML：

- `interactive_01_trajectory_3d.html`
- `interactive_02_lc_scatter.html`
- `interactive_03_ancestry_evolution.html`
- `interactive_04_performance_heatmap.html`

中文字型設定包含 `Microsoft JhengHei`、`SimHei`、`Arial Unicode MS` 與 `DejaVu Sans` fallback，PNG 使用 300 DPI。

為避免 Windows 環境嘗試寫入 `C:\Users\s06t0\.matplotlib` 時發生權限問題，`analysis.visualize` 會在未設定 `MPLCONFIGDIR` 時自動使用 workspace 內的 `.tmp/matplotlib`。

## Report CLI

執行：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m analysis.report
```

常用參數：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m analysis.report --no-interactive
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m analysis.report --result-dir data/results --output-dir data/results/figures
```

輸出內容：

- `phase04_statistics.txt`
- `statistics_run_counts.csv`
- `statistics_h1_correlations.csv`
- `statistics_h2_anova.csv`
- `statistics_h2_tukey.csv`
- `statistics_h3_kruskal.csv`
- `statistics_h3_correlations.csv`
- 8 張 PNG
- 互動 HTML（除非使用 `--no-interactive`）

目前 repo 未包含正式 Phase 3 batch 結果資料；若 `data/results` 沒有 CSV，report 會輸出空統計表與 placeholder 圖。正式分析應在 Phase 3 matrix 完成後再執行。

## Dash Web UI

啟動：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe dashboard/app.py
```

預設位址：

```text
http://127.0.0.1:8050
```

已實作功能：

- Result directory 輸入與 refresh。
- Instance、strategy、generation 篩選。
- Overview tab：3D run trajectory、best fitness convergence、performance heatmap。
- LC/Fitness tab：LC vs fitness scatter。
- Lineage tab：avg LC evolution、upset offspring rate。
- Tables tab：strategy summary table、ANOVA p-value/effect size、CSV 匯出。

## 測試

第一次直接執行 pytest 時，Windows 預設 temp 目錄 `AppData\Local\Temp\pytest-of-s06t0` 權限不足，與 Phase 3 文件記錄的環境問題一致。因此改用 workspace 內 `.tmp/pytest`：

```powershell
New-Item -ItemType Directory -Force -Path .\.tmp\pytest
$env:TMP=(Resolve-Path .\.tmp\pytest).Path
$env:TEMP=$env:TMP
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

結果：

```text
33 passed, 6 warnings in 26.46s
```

Warnings：

- Dash 內建 `dash_table.DataTable` 有 deprecation warning，功能仍可用。
- Seaborn boxplot 觸發 Matplotlib `vert` deprecation warning，功能仍可用。

## Phase 4 Gate 狀態

| Gate | 狀態 |
|---|---|
| H1 LC-fitness correlation | 已實作 |
| H2 ANOVA + Tukey HSD | 已實作 |
| H3 ancestry/diversity 非參數檢定 | 已實作 |
| 8 張 PNG 圖表產生器 | 已實作 |
| Plotly Dash Web UI | 已實作 |
| `python -m analysis.report` | 已實作 |
| Phase 4 focused tests | 已實作並通過 |
