# Phase 3：實驗系統 — 實驗管線 + 並行執行 + 完整實驗

**時程：第 5-6 週**
**前置依賴：Phase 2（GA 引擎 + 血統追蹤完成）**

---

## 目標

建立自動化的實驗執行管線，利用 i7-8750H 的 12 執行緒並行跑實驗，完成所有 **30 runs × 4 策略 × 2 TSP 實例 = 240 次 GA 運行**，並將原始數據存為結構化格式。

---

## 任務清單

### Week 5：實驗管線建立

- [ ] **T3.1** 實作 `experiments/config.py` — 實驗參數設定
  - `ExperimentConfig` dataclass（所有 GA 參數 + 育種策略 + 血統追蹤設定）
  - 預定義四組實驗設定（A/B/C/D）的工廠函數
  - 參數序列化/反序列化（JSON），確保實驗可重現

- [ ] **T3.2** 實作 `experiments/runner.py` — 實驗執行器
  - `run_single(config, seed) → ExperimentResult`：單次實驗封裝
  - `run_batch(config, n_runs, base_seed) → List[ExperimentResult]`：批次執行
  - 自動存檔邏輯：每次 run 完成後即存為 CSV/JSON
  - 進度回報（tqdm 進度條 + 預估剩餘時間）

- [ ] **T3.3** 實作 `experiments/parallel.py` — 多進程並行
  - 使用 `joblib.Parallel(n_jobs=10)` 並行執行（預留 2 緒給系統）
  - 錯誤隔離：單次 run 失敗不影響其他 run
  - 記憶體監控：估算 10 個並行進程的記憶體用量

- [ ] **T3.4** 設計數據輸出格式
  - **世代級數據**（`results/{strategy}_{instance}_gen.csv`）：
    每行 = 一個世代的統計數據（best/avg/worst fitness, diversity, avg_lc, correlation, upset_rate）
  - **個體級數據**（`results/{strategy}_{instance}_ind.csv`）：
    每行 = 一個個體（id, generation, fitness, lc, ancestry_entropy, effective_founders, parent_ids）
    僅每 50 代記錄一次完整快照（節省空間）
  - **彙總數據**（`results/summary.csv`）：
    每行 = 一次 run 的最終結果（strategy, instance, seed, final_best_fitness, gap_to_optimal, convergence_gen）

- [ ] **T3.5** 實作「收斂世代數」計算邏輯
  - 定義：首次達到最優解 ±X% 的世代數
  - 多個門檻（5%, 10%, 15%）同時計算

### Week 6：執行完整實驗 + 除錯

- [ ] **T3.6** 先跑小規模測試：eil51 × tournament × 5 runs
  - 驗證數據輸出格式正確
  - 確認並行執行無 race condition
  - 檢查記憶體用量是否合理

- [ ] **T3.7** 執行完整實驗批次
  - **第一批（eil51）**：4 策略 × 30 runs = 120 次 → 預估 2-3 分鐘
  - **第二批（kroA100）**：4 策略 × 30 runs = 120 次 → 預估 8-12 分鐘
  - 全程記錄 log（時間戳、隨機種子、最終結果）

- [ ] **T3.8** 數據完整性檢查
  - 確認 240 次 run 全部成功完成
  - 確認每次 run 的世代數 = 設定值
  - 確認不同 seed 的結果確實不同（非重複數據）
  - 確認相同 seed 重跑結果完全一致（可重現性）

- [ ] **T3.9** 初步數據探索
  - 用 pandas 快速瀏覽各組統計摘要
  - 確認各策略的 LC 趨勢方向符合預期
  - 發現明顯異常時回頭檢查 GA 引擎邏輯

---

## 交付物

| 檔案 | 狀態 | 驗證方式 |
|---|---|---|
| `experiments/config.py` | 新建 | 四組設定可正確序列化/反序列化 |
| `experiments/runner.py` | 新建 | 單次 + 批次執行正常 |
| `experiments/parallel.py` | 新建 | 10 進程並行無 crash |
| `data/results/` 下所有 CSV | 新建 | 240 次 run 全部有對應檔案 |
| `data/results/summary.csv` | 新建 | 240 行完整彙總 |

---

## Phase 3 完成標準（Gate）

> **必須全部滿足才進入 Phase 4：**

- [x] **里程碑 1**：標準 GA（tournament）在 eil51 上 30 runs 平均差距 < 10%
- [x] **里程碑 4**：eil51 完整實驗（120 runs）在 5 分鐘內完成
- [x] 240 次 run 全部成功，數據檔案完整
- [x] LC 趨勢初步確認：精英組↑、劣質組↓、隨機組 ≈ 中等
- [x] 固定 seed 重跑結果 100% 一致

---

## 效能預估（i7-8750H, 10 並行）

```
eil51:   120 runs ÷ 10 並行 = 12 批次 × ~20s/run ≈ 4 分鐘
kroA100: 120 runs ÷ 10 並行 = 12 批次 × ~75s/run ≈ 15 分鐘
──────────────────────────────────────────────────────────
總計：約 20 分鐘 計算時間
```

記憶體估算：每進程 ~50 MB（距離矩陣 + 族群 + AP 字典）× 10 = ~500 MB，遠低於系統限制。

---

## 風險與注意事項

| 風險 | 對策 |
|---|---|
| 並行寫檔衝突 | 每個 run 寫入獨立檔案，最後合併 |
| 某些 run 因隨機性極差導致異常值 | 30 runs 的統計夠穩健，記錄但不排除 |
| kroA100 單次 run 超時 | 設 300 代上限，不等收斂直接截止 |
| 數據量過大（個體級數據） | 僅每 50 代存一次完整快照，其餘只存統計量 |
