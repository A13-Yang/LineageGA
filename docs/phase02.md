# Phase 2 實作紀錄

日期：2026-06-24

## 實作範圍

Phase 2 已依 `plan/phase02_plan.md` 完成血統追蹤模組、四種育種策略與完整 GA 引擎整合。實作前已閱讀 `docs/phase01.md`，並沿用 Phase 1 已完成的 `Individual`、TSPLIB parser、fitness、crossover、mutation、population 與 selection 基礎介面。

本階段新增的主流程已可在 eil51 上執行含 AP/LC 血統追蹤的完整 GA，並於每代記錄適應度、多樣性、平均 LC、LC-fitness correlation 與逆勢子代率。

## 新增與更新檔案

| 路徑 | 內容 |
|---|---|
| `src/lineage.py` | 新增 `LineageTracker`，實作始祖品質 `q_f`、子代 AP 合併、剪枝正規化、LC、Shannon entropy、有效始祖數 |
| `src/ga_engine.py` | 新增 `GAConfig`、`GAEngine`、`GenerationRecord`、`ExperimentResult`、`IndividualSnapshot` 與逆勢子代偵測 |
| `src/selection.py` | 擴充 `elite`、`poor`、`random` 三種策略；`tournament` 雙親會盡量避免選到同一個體 |
| `src/population.py` | 將 edge-difference diversity 改成 NumPy 矩陣計算，保留原語意但加速 Phase 2 每代紀錄 |
| `tests/test_lineage.py` | 新增血統追蹤單元測試，包含設計書 §3.4 `LC_X = 0.75` |
| `tests/test_ga_engine.py` | 新增 GA 引擎整合測試，涵蓋 50 代 run、菁英保留、四策略、LC 完整性與逆勢子代判斷 |
| `tests/test_population_selection_mutation.py` | 補上 elite/poor/random selection 策略測試 |
| `experiments/phase02_smoke.py` | 新增 eil51 300 代 Phase 2 smoke run，輸出收斂與 LC 趨勢摘要 |

## 血統追蹤實作

`LineageTracker` 使用設計書第 3 節公式：

- 始祖品質：依初始族群 fitness 排名計算 `q_f = 1 - (rank(f) - 1) / (N - 1)`，TSP 為最小化問題，fitness 越小排名越前。
- 子代 AP：使用 50/50 雙親血統混合，合併後剪枝 `< prune_threshold` 的始祖項目。
- 剪枝正規化：預設門檻為 `0.001`，剪枝後重新正規化並檢查總和接近 1.0。
- LC：計算 `LC_i = sum(p_i,f * q_f)`。
- 補充指標：提供 AP Shannon entropy 與 effective founders。

已驗證設計書 §3.4 的 4 始祖手算範例：

```text
AP_X = {f1: 0.5, f2: 0.25, f3: 0.25, f4: 0.0}
q = {f1: 1.0, f2: 2/3, f3: 1/3, f4: 0.0}
LC_X = 0.75
```

## GA 引擎實作

`GAEngine` 的主流程：

1. 建立初始族群並計算 fitness。
2. 依初始 fitness 排名計算 founder quality，並為始祖補上 AP 與 LC。
3. 每代執行 selection、OX/PMX crossover、AP/LC 計算、swap mutation、fitness 評估與 elitism。
4. 交叉後、突變前計算逆勢子代：子代 fitness 小於雙親平均 fitness 即視為逆勢子代。
5. 每代建立 `GenerationRecord`，最後回傳 `ExperimentResult`。

目前 `GAConfig` 支援的 Phase 2 參數包含 population size、generation count、crossover type、mutation rate、elite count、selection strategy、tournament size、elite/poor pool ratio、AP 剪枝門檻與 snapshot interval。

## 驗證結果

使用 Phase 1 建立的 `bloodline-ga` 環境執行：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m pytest tests -q
```

結果：

```text
25 passed in 8.13s
```

編譯檢查：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m py_compile src\lineage.py src\ga_engine.py src\selection.py src\population.py experiments\phase02_smoke.py tests\test_lineage.py tests\test_ga_engine.py
```

結果：通過。

Phase 2 smoke run：

```powershell
C:\Users\s06t0\anaconda3\envs\bloodline-ga\python.exe -B -m experiments.phase02_smoke
```

設定：eil51、seed 42、population 120、300 generations、OX、mutation rate 0.08、2 elites、tournament size 5。

結果：

```text
instance=eil51
seed=42
optimal_length=426
initial_best=1386
final_best=483
gap_to_optimal=13.38%
initial_avg_lc=0.5000
final_avg_lc=0.9119
elapsed_time=25.87s
generation,best,avg_lc,diversity,upset_rate
0,1386,0.5000,0.9596,0.0000
50,664,0.9119,0.3633,0.1441
100,572,0.9119,0.1765,0.1441
150,552,0.9119,0.1383,0.1271
200,511,0.9119,0.1686,0.1102
250,499,0.9119,0.1442,0.0847
300,483,0.9119,0.1342,0.0763
```

## Gate 對照

| Gate | 狀態 | 驗證 |
|---|---|---|
| 設計書 §3.4 精確重現 `LC_X = 0.75` | 通過 | `tests/test_lineage.py` |
| 四種育種策略皆能獨立執行完整 GA run | 通過 | `tests/test_ga_engine.py` parametrized test |
| 每個個體都有正確 `ancestry` 與 `lc` | 通過 | final population LC/AP 測試 |
| 逆勢子代偵測邏輯正確 | 通過 | 手動案例單元測試 |
| 標準 GA 在 eil51 上 300 代收斂至最優解 ±15% 以內 | 通過 | final best 483，距 426 為 13.38% |

## 注意事項

- `docs/phase01.md` 提到 Phase 2/3 分工時曾寫「Phase 3 再加入 AP/LC」；本次依 `plan/phase02_plan.md` 的最新版任務，已在 Phase 2 先完成 AP/LC 與 GA engine。
- 本資料夾目前不是 Git worktree，無法用 `git status` 或 `git diff` 產生版本差異摘要。
- Phase 2 smoke 是單一 seed 的功能與 gate 驗證；後續 Phase 3/4 若要做研究結論，仍需使用多 seed、多策略批次實驗與統計檢定。
