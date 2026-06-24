# Phase 5 實作紀錄
日期：2026-06-24

## 前置閱讀

本階段實作前已閱讀：

- `docs/phase01.md`：確認 TSPLIB parser、TSP fitness、permutation crossover/mutation、population 與基礎測試。
- `docs/phase02.md`：確認 `GAEngine`、`GAConfig`、AP/LC lineage tracking、四種 selection strategy 與 upset offspring 定義。
- `docs/phase03.md`：確認 `ExperimentConfig`、runner、parallel batch、CSV/JSON artifact schema 與完整性檢查。
- `docs/phase04.md`：確認統計分析、PNG/Plotly 視覺化、Dash Web UI 與 report CLI 的資料讀取路徑。

## 實作範圍

Phase 5 已將原本以 TSP permutation 為核心的系統，擴充為同一套 lineage-aware GA 可支援三種問題：

- `tsp`：沿用既有 TSPLIB tour length，最小化。
- `nk`：新增 NK Landscape，bitstring 編碼，最大化 `[0, 1]` fitness。
- `trap`：新增 concatenated deceptive Trap Function，bitstring 編碼，最大化 `[0, 1]` fitness。

本回合完成程式、分析、Dashboard 與測試驗證；完整 600 次 NK + 240 次 Trap 正式批次資料尚未在本回合執行，以避免實作驗證階段耗費過長計算時間。已提供 `run_full_phase05()` 作為正式批次入口。

## 新增與修改檔案

| 檔案 | 內容 |
|---|---|
| `src/nk_landscape.py` | 新增 `NKLandscape`，支援 `n`、`k`、seeded contribution tables、fitness 與小規模 brute-force optimum |
| `src/trap_function.py` | 新增 `TrapFunction` 與 `trap_fitness`，支援 concatenated deceptive trap |
| `src/crossover.py` | 新增 `uniform_crossover` 與 `uniform_crossover_pair` |
| `src/mutation.py` | 新增 `bitflip_mutation` |
| `src/population.py` | 新增 bitstring founder population、Hamming diversity，並讓 elitism 支援 maximization |
| `src/selection.py` | tournament / elite / poor selection 支援 maximization |
| `src/lineage.py` | founder quality 排名支援 minimization / maximization |
| `src/ga_engine.py` | `GAConfig` 新增 `problem_type`、`chromosome_length`、`fitness_function`、`maximize`；`GAEngine` 支援 tsp/nk/trap |
| `experiments/config.py` | `ExperimentConfig` 新增 `problem_type`、`problem_params`，並提供 Phase 5 NK/Trap config matrix |
| `experiments/runner.py` | CSV schema 新增 problem metadata，convergence/gap 支援最大化問題 |
| `experiments/parallel.py` | 新增 `run_full_phase05()`，memory estimate 支援 bitstring 問題 |
| `analysis/statistics.py` | 新增 Phase 5 統計表：NK K effect、problem-family LC effect comparison |
| `analysis/visualize.py` | 新增 `fig09_nk_k_effect.png`、`fig10_problem_lc_effect_comparison.png` 與對應 Plotly 圖 |
| `analysis/report.py` | report CLI 會輸出 Phase 5 CSV tables |
| `dashboard/app.py` | 新增 `Extended` tab，展示 NK K effect 與 TSP/NK/Trap LC effect comparison |
| `tests/test_phase05_extended.py` | 新增 Phase 5 focused tests |
| `README.md` | 更新完整專案說明、安裝、快速開始、完整實驗、Web UI 與目錄結構 |

## 核心設計

`GAEngine` 現在以 problem adapter 的方式運作：

- TSP：由 `distance_matrix` 評估 tour length，`maximize=False`，使用 OX/PMX 與 swap mutation。
- NK/Trap：由 `fitness_function` 評估 bitstring，`maximize=True`，使用 uniform crossover 與 bitflip mutation。
- AP/LC lineage tracking 保持不依賴編碼方式；差異只在 founder quality 的排序方向。
- `GenerationRecord.best_fitness` 代表該問題方向下的最佳值：TSP 越小越好，NK/Trap 越大越好。

## Phase 5 實驗入口

NK matrix：

```python
from experiments.config import predefined_nk_configs
from experiments.parallel import run_parallel

configs = predefined_nk_configs(n=20, k_values=(0, 2, 5, 10, 19))
statuses = run_parallel(configs, n_runs=30, base_seed=5000, n_jobs=10)
```

Trap matrix：

```python
from experiments.config import predefined_trap_configs
from experiments.parallel import run_parallel

configs = predefined_trap_configs(block_size=5, n_blocks_values=(5, 10))
statuses = run_parallel(configs, n_runs=30, base_seed=6000, n_jobs=10)
```

完整 Phase 5：

```python
from experiments.parallel import run_full_phase05

statuses = run_full_phase05(n_runs=30, base_seed=5000, n_jobs=10)
```

## 分析與視覺化

新增的 Phase 5 分析重點：

- `nk_k_lc_fitness_correlations(data)`：依 NK `K` 與 strategy 計算 LC-fitness Pearson/Spearman correlation。
- `problem_lc_effect_comparison(data)`：比較 `tsp`、`nk`、`trap` 的 LC-fitness correlation。
- `fig09_nk_k_effect.png`：K 值 vs. LC-fitness correlation。
- `fig10_problem_lc_effect_comparison.png`：TSP/NK/Trap 的 LC effect 比較。
- Dash `Extended` tab：互動式展示上述兩個延伸圖表。

## 驗證結果

使用 `bloodline-ga` Conda 環境與 workspace 內 `.tmp/pytest` 暫存目錄執行：

```powershell
$env:TMP=(Resolve-Path .\.tmp\pytest).Path
$env:TEMP=$env:TMP
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

結果：

```text
40 passed, 6 warnings in 26.11s
```

Warnings 與 Phase 4 相同，來自 Dash `dash_table.DataTable` deprecation 與 Seaborn/Matplotlib boxplot deprecation，不影響功能。

## Gate 狀態

| Gate | 狀態 |
|---|---|
| `src/nk_landscape.py` | 已實作並測試 |
| `src/trap_function.py` | 已實作並測試 |
| uniform crossover / bitflip mutation | 已實作並測試 |
| `GAEngine` 支援 tsp/nk/trap | 已實作並測試 |
| Phase 5 config matrix | 已實作，`predefined_phase05_configs()` 產生 28 個 config cell |
| 延伸統計分析 | 已實作並測試 |
| 延伸 PNG / Plotly 圖表 | 已實作並測試 |
| Dashboard Extended tab | 已實作並由 smoke test 覆蓋 |
| 完整 840-run 正式資料 | 尚未執行；已提供 `run_full_phase05()` |
| 全部單元測試 | 通過 |
