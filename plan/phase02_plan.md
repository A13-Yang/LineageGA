# Phase 2：核心引擎 — 血統追蹤模組 + GA 引擎整合

**時程：第 3-4 週**
**前置依賴：Phase 1（GA 基本元件全部完成）**

---

## 目標

實作本研究的**核心創新模組 — 血統追蹤（Lineage Tracking）**，並將所有 GA 元件整合為完整的 `GAEngine`。Phase 2 結束時，應能跑一次含血統追蹤的完整 GA，並輸出每個個體的 LC 值。

---

## 任務清單

### Week 3：血統追蹤模組（研究核心）

- [ ] **T2.1** 實作 `src/lineage.py` — `LineageTracker` 類別
  - `compute_founder_quality(population)` → 始祖品質分數 q_f
    - 公式：`q_f = 1 - (rank(f) - 1) / (N - 1)`
  - `compute_offspring_ancestry(parent_a, parent_b)` → 子代 AP 字典
    - 公式：`p_child,f = 0.5 * p_a,f + 0.5 * p_b,f`
    - 合併雙親字典 → 剪枝（< threshold）→ 重新正規化
  - `compute_lc(ancestry)` → 血統濃度
    - 公式：`LC_i = Σ_f p_i,f × q_f`
  - `compute_ancestry_entropy(ancestry)` → Shannon entropy（補充指標）
  - `compute_effective_founders(ancestry)` → 有效始祖數（補充指標）

- [ ] **T2.2** 實作剪枝機制
  - 門檻值預設 `0.001`
  - 剪枝後重新正規化確保 `Σ p_i,f = 1.0`
  - 效能測試：比較剪枝前後的 AP 字典大小與計算時間

- [ ] **T2.3** 撰寫 `tests/test_lineage.py` — 血統追蹤單元測試
  - **設計書範例驗證（§3.4）**：4 個始祖的手動範例，驗證 `LC_X = 0.75`
  - AP 合併後 `Σ p_i,f = 1.0`（浮點誤差容忍 1e-10）
  - 剪枝後重新正規化正確性
  - 始祖個體的 AP = `{self.id: 1.0}`
  - 多代遞推測試（3-5 代鏈式交配，驗證 AP 遞推正確）

- [ ] **T2.4** 擴充 `src/selection.py` — 新增三種育種策略
  - `elite`：從 fitness 排名 top 20% 中選取
  - `poor`：從 fitness 排名 bottom 20% 中選取
  - `random`：完全隨機選取

### Week 4：GA 引擎整合

- [ ] **T2.5** 實作 `src/ga_engine.py` — `GAEngine` 類別
  - 整合：population → selection → crossover → lineage → mutation → fitness → elitism
  - 主迴圈流程：
    1. 初始化族群（始祖），計算 q_f
    2. 每代：選擇 → 交叉 → 計算子代 AP & LC → 突變 → 評估適應度 → 菁英保留
    3. 每代記錄 `GenerationRecord`
  - 回傳 `ExperimentResult`（包含完整的世代歷史）

- [ ] **T2.6** 定義數據結構
  - `GenerationRecord`：generation, best/avg/worst fitness, diversity, avg_lc, lc_fitness_correlation, upset_offspring_rate
  - `ExperimentResult`：config, seed, history, final_best_individual, elapsed_time
  - `IndividualSnapshot`：簡化版個體記錄（每 N 代完整快照一次）

- [ ] **T2.7** 實作「逆勢子代」偵測邏輯
  - 定義：子代適應度優於 `(parent_a.fitness + parent_b.fitness) / 2`
  - 注意：TSP 是最小化問題，「優於」= 路徑更短
  - 每代統計出現率

- [ ] **T2.8** 撰寫 `tests/test_ga_engine.py` — 引擎整合測試
  - GA 完整 run（50 代）不 crash
  - 菁英保留確實保留了上一代的最佳個體
  - 四種育種策略皆能正常執行
  - 血統追蹤數據完整（每個個體都有 LC 值）

- [ ] **T2.9** 初步驗證：標準 GA（tournament）在 eil51 上跑 300 代
  - 確認收斂趨勢合理
  - 確認 LC 值隨世代變化的趨勢
  - 輸出簡單的收斂曲線（print or 簡易 matplotlib）

---

## 交付物

| 檔案 | 狀態 | 驗證方式 |
|---|---|---|
| `src/lineage.py` | 新建 | 設計書 §3.4 範例 LC = 0.75 |
| `src/ga_engine.py` | 新建 | eil51 上完整 300 代 run 成功 |
| `src/selection.py` | 修改 | 新增 elite/poor/random 三種策略 |
| `tests/test_lineage.py` | 新建 | `pytest` 全部通過 |
| `tests/test_ga_engine.py` | 新建 | `pytest` 全部通過 |

---

## Phase 2 完成標準（Gate）

> **必須全部滿足才進入 Phase 3：**

- [x] **里程碑 2**：設計書範例 §3.4 精確重現 `LC_X = 0.75`
- [x] 四種育種策略（elite/tournament/poor/random）皆能獨立執行完整 GA run
- [x] 每個個體都有正確的 `ancestry` 字典與 `lc` 值
- [x] 逆勢子代偵測邏輯正確（手動驗證 3-5 個案例）
- [x] 標準 GA 在 eil51 上 300 代收斂至最優解 ±15% 以內

---

## 風險與注意事項

| 風險 | 對策 |
|---|---|
| AP 字典浮點累積誤差 | 每次計算後 assert `abs(sum(ap.values()) - 1.0) < 1e-9` |
| 剪枝門檻太高導致 LC 計算不準確 | 先用 0.001，若發現影響超過 1% 則降為 0.0001 |
| GA 引擎整合時模組介面不一致 | Phase 1 已定義統一介面，整合時逐模組替換測試 |
| 逆勢子代定義：突變後適應度改變 | 逆勢判斷在交叉後、突變前進行，避免突變干擾 |
