# Phase 1 實作紀錄

日期：2026-06-24

## 實作範圍

Phase 1 已完成基礎專案骨架、TSPLIB 資料層與標準 GA 的底層元件。此階段尚未加入血統追蹤、完整實驗管線或統計視覺化，保留給後續 phase。

## 新增與更新檔案

| 路徑 | 內容 |
|---|---|
| `pyproject.toml` | 定義專案 metadata、Python 3.11+、核心依賴與 dev 測試依賴 |
| `environment.yml` | 可重建 `bloodline-ga` Conda 環境 |
| `README.md` | Phase 1 快速啟動方式 |
| `src/__init__.py` | 核心套件初始化 |
| `src/tsplib_parser.py` | 解析 TSPLIB `.tsp` EUC_2D 與 `.opt.tour` |
| `src/fitness.py` | 建立距離矩陣與計算封閉 TSP tour 長度 |
| `src/individual.py` | `Individual` dataclass、全域 ID 產生器、建立輔助函式 |
| `src/crossover.py` | Order Crossover (OX)、Partially Mapped Crossover (PMX)、排列合法性檢查 |
| `src/mutation.py` | Swap mutation 與個體突變輔助函式 |
| `src/selection.py` | Tournament selection 與 `select_parents` 統一介面 |
| `src/population.py` | 初始族群、菁英保留與 edge-difference 多樣性 |
| `experiments/__init__.py` | 實驗套件初始化 |
| `experiments/phase01_smoke.py` | 無血統追蹤的簡單 eil51 GA smoke run |
| `analysis/__init__.py` | 分析套件初始化 |
| `tests/test_crossover.py` | OX/PMX 100 次隨機合法排列測試 |
| `tests/test_fitness.py` | 手工距離與 eil51 最優 tour 長度測試 |
| `tests/test_tsplib_parser.py` | eil51 / kroA100 TSPLIB 解析與最優 tour 長度測試 |
| `tests/test_population_selection_mutation.py` | 初始族群、swap mutation、elitism、selection、diversity 測試 |
| `data/tsplib/eil51.tsp` | TSPLIB eil51 實例 |
| `data/tsplib/eil51.opt.tour` | TSPLIB eil51 最優 tour |
| `data/tsplib/kroA100.tsp` | TSPLIB kroA100 實例 |
| `data/tsplib/kroA100.opt.tour` | TSPLIB kroA100 最優 tour |

## Conda 環境

已建立額外 Conda 環境：

```powershell
conda activate bloodline-ga
```

實際位置：

```text
C:\Users\s06t0\anaconda3\envs\bloodline-ga
```

環境使用 Python 3.11，並已透過下列方式安裝專案與 dev 依賴：

```powershell
python -m pip install -e .[dev]
```

若要重建環境，可在專案根目錄執行：

```powershell
conda env create -f environment.yml
```

## TSPLIB 資料

原本先嘗試連線官方 TSPLIB95 站台，但連線逾時。實際資料改由公開 TSPLIB 鏡像下載：

```text
https://raw.githubusercontent.com/coin-or/jorlib/master/jorlib-core/src/test/resources/tspLib/tsp/
```

已驗證：

| 實例 | 城市數 | 最優 tour 長度 |
|---|---:|---:|
| `eil51` | 51 | 426 |
| `kroA100` | 100 | 21282 |

## 驗證結果

使用新建的 `bloodline-ga` 環境執行：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

結果：

```text
10 passed in 3.13s
```

執行 Phase 1 smoke run：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m experiments.phase01_smoke
```

結果：

```text
initial_best=1386
final_best=747
improved=True
```

另外已驗證所有 Phase 1 依賴可 import：

```text
imports ok
```

Matplotlib 第一次 import 時嘗試寫入使用者家目錄字型快取，在目前沙盒下出現權限提示；這不影響 import 或測試。若需要完全安靜，可在執行繪圖前將 `MPLCONFIGDIR` 指到專案內可寫目錄。

## Gate 對照

| Gate | 狀態 |
|---|---|
| `src/` 模組可 import | 通過 |
| `eil51.tsp` 解析為 51 個城市座標 | 通過 |
| `eil51` 最優 tour 長度為 426 | 通過 |
| `kroA100` 最優 tour 長度為 21282 | 通過 |
| OX/PMX 交叉 100 次測試零失敗 | 通過 |
| 無血統追蹤簡單 GA 在 eil51 上 100 代呈下降趨勢 | 通過 |

## 後續銜接

Phase 2 可直接在現有元件上加入完整 GA engine、實驗 config 與更細的策略選擇；Phase 3 再加入 AP/LC 血統追蹤，不需要改動目前的 TSPLIB、fitness、crossover、mutation 與 population 基礎介面。
