# Phase 3 實作紀錄

日期：2026-06-24

## 實作範圍

本階段已依 `plan/phase03_plan.md` 建立實驗管線、批次執行、平行執行、CSV/JSON 輸出、收斂世代計算、結果完整性檢查與初步 pandas 彙整工具。實作前已閱讀 `docs/phase01.md` 與 `docs/phase02.md`，並沿用 Phase 2 已完成的 `GAEngine`、`GAConfig`、`ExperimentResult`、`GenerationRecord` 與 AP/LC 血統追蹤資料。

本次重點是完成可重現的實驗執行基礎設施；未在本回合直接執行完整 240 runs，以避免在實作驗證階段耗費過長計算時間。

## 新增與更新檔案

| 路徑 | 內容 |
|---|---|
| `experiments/config.py` | 新增 `ExperimentConfig`，支援 JSON 序列化/反序列化、TSPLIB 路徑解析、GAConfig 轉換，以及 A/B/C/D 四組策略工廠函式 |
| `experiments/runner.py` | 新增 `run_single`、`run_batch`、per-run CSV/JSON 存檔、aggregate CSV 合併、收斂世代計算、完整性檢查與 pandas 摘要工具 |
| `experiments/parallel.py` | 新增 joblib 平行執行、錯誤隔離 `RunStatus`、完整 Phase 3 matrix launcher 與平行記憶體估算 |
| `tests/test_phase03_pipeline.py` | 新增 Phase 3 設定、runner、parallel、收斂計算與完整性檢查測試 |
| `docs/phase03.md` | 本文件，記錄 Phase 3 實際實作內容與驗證方式 |

## 實驗設定

`ExperimentConfig` 是可 JSON 化的實驗設定層，保存 instance、策略、GA 參數、血統追蹤參數、snapshot interval、收斂門檻與輸出目錄。

預設四組策略：

| 代碼 | strategy | selection_strategy |
|---|---|---|
| A | `tournament` | `tournament` |
| B | `elite` | `elite` |
| C | `poor` | `poor` |
| D | `random` | `random` |

`predefined_configs()` 會展開為 4 策略 × 2 實例（`eil51`、`kroA100`），對應完整 Phase 3 的 8 個實驗 cell。

## 輸出格式

為避免平行寫檔衝突，每個 run 會先寫入獨立檔案：

```text
data/results/runs/{strategy}_{instance}/{strategy}_{instance}_seed{seed}_gen.csv
data/results/runs/{strategy}_{instance}/{strategy}_{instance}_seed{seed}_ind.csv
data/results/runs/{strategy}_{instance}/{strategy}_{instance}_seed{seed}_summary.csv
data/results/runs/{strategy}_{instance}/{strategy}_{instance}_seed{seed}_summary.json
```

批次完成後再合併成計畫指定的總表：

```text
data/results/{strategy}_{instance}_gen.csv
data/results/{strategy}_{instance}_ind.csv
data/results/summary.csv
```

世代級資料包含 best/avg/worst fitness、diversity、avg_lc、LC-fitness correlation、upset offspring rate 與 best individual id。

個體級資料只輸出 GAEngine snapshot 中的完整個體快照；Phase 3 預設 `snapshot_interval=50`，並保留 final generation snapshot。欄位包含 id、generation、fitness、lc、ancestry entropy、effective founders、ancestry size 與 parent ids。

彙總資料包含每次 run 的最終 fitness、optimal length、gap、5%/10%/15% 收斂世代、最終 LC/diversity、elapsed time 與 GA 參數。

## 執行方式

小規模單一設定：

```python
from experiments.config import config_a
from experiments.runner import run_batch

config = config_a(instance="eil51")
run_batch(config, n_runs=5, base_seed=1000)
```

完整 Phase 3 matrix：

```python
from experiments.parallel import run_full_phase03

statuses = run_full_phase03(n_runs=30, base_seed=1000, n_jobs=10)
```

結果完整性檢查：

```python
from experiments.config import predefined_configs
from experiments.runner import validate_result_integrity

report = validate_result_integrity(
    predefined_configs(),
    expected_runs_per_config=30,
)
```

初步 pandas 摘要：

```python
from experiments.runner import summarize_results

summary = summarize_results("data/results")
```

## 驗證結果

使用 `bloodline-ga` Conda 環境驗證。因目前沙盒不允許 pytest 使用預設 `AppData\\Local\\Temp`，測試時先將暫存目錄指向專案內 `.tmp/pytest`。

```powershell
New-Item -ItemType Directory -Force -Path .\.tmp\pytest
$env:TMP=(Resolve-Path .\.tmp\pytest)
$env:TEMP=$env:TMP
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

結果：

```text
30 passed in 4.04s
```

另外曾直接執行 `python -m pytest tests -q`，但 shell 預設 Python 為 `C:\msys64\mingw64\bin\python.exe`，未安裝 pytest，因此不作為專案失敗。

## Gate 對照

| Gate | 狀態 | 備註 |
|---|---|---|
| `experiments/config.py` 可建立四組設定並 JSON round-trip | 通過 | `tests/test_phase03_pipeline.py` |
| `experiments/runner.py` 單次與批次執行正常 | 通過 | 小型 eil51 測試會產生 per-run 與 aggregate CSV/JSON |
| `experiments/parallel.py` 平行 runner 具錯誤隔離 | 通過 | 測試以 `n_jobs=1` 驗證 orchestration，正式可設 `n_jobs=10` |
| 收斂世代 5%/10%/15% 計算 | 通過 | 使用手工 `GenerationRecord` 驗證 |
| 數據完整性檢查 | 通過 | 檢查 summary rows、unique seeds、aggregate file 與每 seed 世代數 |
| 240 runs 完整實驗資料 | 未執行 | 管線已提供 `run_full_phase03(n_runs=30, n_jobs=10)` 啟動完整批次 |

## 注意事項

- `run_single` 會重建 GAConfig 並載入 TSPLIB distance matrix，確保每個 job process 彼此隔離。
- per-run 檔案先落在 `data/results/runs/`，可避免多進程同時 append 同一總表造成 race condition。
- `merge_run_artifacts` 可重跑，會從 per-run CSV 重新生成 aggregate CSV 與 `summary.csv`。
- `validate_result_integrity` 目前檢查資料完整性與 seed 唯一性；「相同 seed 重跑 100% 一致」仍建議在正式完整實驗後用固定輸出目錄之外的對照 run 額外比對。
