# LineageGA：基因演算法血統濃度與子代品質研究

本專案用 genetic algorithm 研究「血統濃度」（Lineage Concentration, LC）與子代品質之間的關係。系統從 TSP permutation GA 出發，加入 AP/LC 血統追蹤、批次實驗、統計分析、視覺化與 Dash Web UI，並在 Phase 5 擴充至 NK Landscape 與 Trap Function，以觀察 LC 效應在不同 fitness landscape 上的適用邊界。

## 快速分析與發現

您可以直接透過腳本執行四種育種策略（Tournament, Elite, Poor, Random）的快速實驗並檢視分析結果：

```powershell
conda activate bloodline-ga
python scripts/run_analysis.py
```

### 實驗核心發現

1. **劣質育種的「逆勢奇蹟」**：在純劣質選擇壓力下（只用最差 20% 交配），逆勢子代率高達 **90%** 且未衰減，反觀精英育種的逆勢率最終降至 0%（早熟收斂，喪失探索能力）。
2. **LC-Fitness 強負相關**：劣質育種組展現極強的相關性（`r = -0.89`）。意味著在逆境中，**血統依然記得誰是好祖先**，LC 成為強烈的適應度預測指標。
3. **始祖命運提早定型**：各策略在約 17%（第 50 代）的演化進程時，LC 就完全「鎖死」不再變動。
4. **精英育種不保證最佳解**：精英育種雖然收斂最快，但多樣性嚴重崩潰，導致解的穩定度（變異度）是錦標賽選擇（Tournament）的兩倍。
5. **逆勢不等於優秀**：劣質育種即便有 90% 的高逆勢率，其找到的歷史最佳解，依舊比不過第 0 代隨機產生的最佳個體。

## 環境安裝

```powershell
conda env create -f environment.yml
conda activate bloodline-ga
python -m pip install -e .[dev]
```

既有環境已建立於：

```text
C:\Users\s06t0\anaconda3\envs\bloodline-ga
```

## 測試與執行

執行測試：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

若 Windows temp 目錄權限受限，可先把 pytest 暫存目錄指到 workspace：

```powershell
New-Item -ItemType Directory -Force -Path .\.tmp\pytest
$env:TMP=(Resolve-Path .\.tmp\pytest).Path
$env:TEMP=$env:TMP
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

執行單次 TSP smoke run：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m experiments.phase02_smoke
```

## 完整實驗

Phase 3 TSP matrix（4 策略 × 2 TSPLIB instances × 30 runs）：

```python
from experiments.parallel import run_full_phase03

statuses = run_full_phase03(n_runs=30, base_seed=1000, n_jobs=10)
```

Phase 5 extension matrix（NK：5 個 K 值 × 4 策略 × 30 runs；Trap：2 個規模 × 4 策略 × 30 runs）：

```python
from experiments.parallel import run_full_phase05

statuses = run_full_phase05(n_runs=30, base_seed=5000, n_jobs=10)
```

輸出會寫入：

```text
data/results/
data/results/runs/
```

## 分析與圖表

產生統計表、PNG 圖與 Plotly HTML：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m analysis.report
```

只產生 CSV 與 PNG、跳過互動 HTML：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m analysis.report --no-interactive
```

Phase 4/5 圖表會輸出到：

```text
data/results/figures/
```

其中 Phase 5 新增：

- `fig09_nk_k_effect.png`
- `fig10_problem_lc_effect_comparison.png`
- `phase05_nk_k_lc_fitness.csv`
- `phase05_problem_lc_effect.csv`

## Web UI

啟動 Dash dashboard：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe dashboard/app.py
```

預設網址：

```text
http://127.0.0.1:8050
```

Dashboard 包含 Overview、LC/Fitness、Lineage、Tables 與 Phase 5 `Extended` tab。

## 專案結構

```text
.
├── README.md
├── pyproject.toml
├── environment.yml
├── docs/
│   ├── phase01.md
│   ├── phase02.md
│   ├── phase03.md
│   ├── phase04.md
│   └── phase05.md
├── plan/
│   ├── phase01_plan.md
│   ├── phase02_plan.md
│   ├── phase03_plan.md
│   ├── phase04_plan.md
│   └── phase05_plan.md
├── src/
│   ├── crossover.py
│   ├── fitness.py
│   ├── ga_engine.py
│   ├── individual.py
│   ├── lineage.py
│   ├── mutation.py
│   ├── nk_landscape.py
│   ├── population.py
│   ├── selection.py
│   ├── trap_function.py
│   └── tsplib_parser.py
├── scripts/
│   └── run_analysis.py
├── experiments/
│   ├── config.py
│   ├── parallel.py
│   └── runner.py
├── analysis/
│   ├── report.py
│   ├── statistics.py
│   └── visualize.py
├── dashboard/
│   └── app.py
├── data/
│   ├── tsplib/
│   └── results/
└── tests/
```

## 實作紀錄

每個階段的實際內容記錄於 `docs/phase01.md` 到 `docs/phase05.md`。Phase 5 已完成程式與小規模自動測試驗證；完整 840-run 延伸實驗可用 `run_full_phase05()` 啟動。
