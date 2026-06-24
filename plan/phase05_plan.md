# Phase 5：延伸與收尾 — NK Landscape + Trap Function + 最終整合

**時程：第 9-10 週**
**前置依賴：Phase 4（統計分析 + 視覺化 + Web UI 完成）**

---

## 目標

1. 實作延伸實驗問題（NK Landscape + Trap Function），驗證「血統濃度效應的適用邊界」
2. 在延伸問題上重跑完整實驗
3. 將延伸結果整合至統計分析與 Web UI
4. 最終系統收尾、文件撰寫

---

## 任務清單

### Week 9：延伸問題實作 + 實驗

- [ ] **T5.1** 實作 `src/nk_landscape.py` — NK Landscape 問題
  - NK Landscape 產生器：可調參數 N（基因長度）和 K（epistasis 強度）
  - K=0：無基因交互作用，適應度地景平滑可分解
  - K=N-1：最大 epistasis，適應度地景近似隨機
  - 建議測試組合：N=20, K ∈ {0, 2, 5, 10, 19}
  - 適應度函數：`fitness(bitstring) → float ∈ [0, 1]`

- [ ] **T5.2** 實作 `src/trap_function.py` — Trap Function 問題
  - 標準 Trap function：具有欺騙性的適應度地景
  - 全域最優在全 1，但區域梯度引導向全 0
  - Concatenated trap（多個 trap block 串接）增加問題規模

- [ ] **T5.3** 適配 GA 引擎以支援二元編碼
  - NK/Trap 使用 bitstring 編碼（非排列），需要：
    - 新增 `uniform_crossover(parent_a, parent_b)` — 均勻交叉
    - 新增 `bitflip_mutation(individual, p)` — 位元翻轉突變
  - `GAEngine` 增加 `problem_type` 參數（"tsp" / "nk" / "trap"）
  - 血統追蹤模組不需修改（AP/LC 與編碼方式無關）

- [ ] **T5.4** 執行 NK Landscape 延伸實驗
  - 5 個 K 值 × 4 策略 × 30 runs = 600 次 run
  - 預估時間：NK fitness 計算比 TSP 快很多，~10 分鐘
  - 重點觀察：H1（LC vs. fitness 相關性）如何隨 K 增大而變化

- [ ] **T5.5** 執行 Trap Function 延伸實驗
  - 2 個規模（trap-5, trap-10）× 4 策略 × 30 runs = 240 次 run
  - 重點觀察：在欺騙性問題上，LC 是否仍能預測子代品質

### Week 10：延伸分析 + 系統收尾

- [ ] **T5.6** 延伸實驗統計分析
  - NK Landscape：計算每個 K 值下的 LC-fitness 相關係數
  - 繪製「K 值 vs. LC-fitness 相關係數」趨勢圖（關鍵圖表）
  - 驗證：相關性是否隨 epistasis 增加而減弱？

- [ ] **T5.7** 延伸結果整合至視覺化
  - 新增圖表：NK Landscape 的 K 值效應圖
  - 新增圖表：TSP vs. NK vs. Trap 的 LC 效應比較
  - 更新 Web UI 儀表板：新增「延伸實驗」頁面

- [ ] **T5.8** 系統整體除錯與效能優化
  - 全流程端到端測試：`實驗設定 → 執行 → 數據 → 分析 → 視覺化 → Web UI`
  - 檢查記憶體洩漏（長時間 run 的記憶體變化）
  - 最佳化瓶頸（profiling 確認無不必要的計算）

- [ ] **T5.9** 撰寫 `README.md`
  - 專案簡介與研究目的
  - 安裝與環境設定指南
  - 快速開始：如何跑一次實驗
  - 完整實驗：如何重現所有結果
  - Web UI 啟動方式
  - 目錄結構說明

- [ ] **T5.10** 最終驗證清單
  - 所有單元測試通過：`pytest tests/ -v`
  - 所有里程碑達成（見下方）
  - Web UI 儀表板所有頁面正常運作
  - `README.md` 中的指令全部可執行

---

## 交付物

| 檔案 | 狀態 | 驗證方式 |
|---|---|---|
| `src/nk_landscape.py` | 新建 | K=0 時 GA 可快速收斂至最優 |
| `src/trap_function.py` | 新建 | GA 在 trap function 上展現欺騙性（收斂至次優） |
| `src/crossover.py` | 修改 | 新增 uniform_crossover |
| `src/mutation.py` | 修改 | 新增 bitflip_mutation |
| `src/ga_engine.py` | 修改 | 支援 tsp/nk/trap 三種問題類型 |
| `analysis/visualize.py` | 修改 | 新增延伸實驗圖表 |
| `dashboard/app.py` | 修改 | 新增延伸實驗頁面 |
| `README.md` | 新建 | 完整專案文件 |
| 延伸實驗數據（CSV） | 新建 | 840 次 run 全部完成 |

---

## Phase 5 完成標準（最終 Gate）

> **專案完成的最終驗證：**

- [x] **里程碑 1**：標準 GA 在 eil51 上 30 runs 平均差距 < 10%
- [x] **里程碑 2**：設計書範例 §3.4 精確重現 `LC_X = 0.75`
- [x] **里程碑 3**：四種育種策略的 LC 趨勢符合預期方向
- [x] **里程碑 4**：eil51 完整實驗在 5 分鐘內完成
- [x] **里程碑 5**（新）：NK Landscape K 值效應圖呈現清晰趨勢
- [x] **里程碑 6**（新）：Web UI 儀表板含 TSP + 延伸實驗，所有頁面正常
- [x] 全部單元測試通過
- [x] `README.md` 中所有指令可成功執行

---

## 風險與注意事項

| 風險 | 對策 |
|---|---|
| NK Landscape 的 AP/LC 行為與 TSP 差異太大 | 這本身就是研究發現，如實報告 |
| 二元編碼的均勻交叉 AP 權重不再是 50/50 | 仍用 0.5/0.5 假設（與設計書一致），論文中討論此限制 |
| 延伸實驗結果推翻 H1 | 不影響研究價值——確認「適用邊界」本身就是貢獻 |
| Web UI 同時展示 TSP + NK + Trap 數據導致架構複雜 | 用 Dash 的多頁面架構（Multi-page app）分隔 |

---

## 專案總產出清單

完成 Phase 5 後，整個專案包含：

```
驗血基因演算法/
├── README.md                           # 完整專案文件
├── pyproject.toml                      # 依賴管理
├── 血統濃度與GA子代品質_實驗設計書.md   # 原始設計書
├── plan/                               # Phase 計畫（5 份）
│
├── src/                                # 核心程式碼（11 個模組）
│   ├── individual.py, population.py
│   ├── crossover.py, mutation.py, selection.py
│   ├── fitness.py, lineage.py, ga_engine.py
│   ├── tsplib_parser.py
│   ├── nk_landscape.py, trap_function.py
│
├── experiments/                        # 實驗管線（3 個模組）
│   ├── config.py, runner.py, parallel.py
│
├── analysis/                           # 分析模組（3 個模組）
│   ├── statistics.py, visualize.py, report.py
│
├── dashboard/                          # Web UI（1 個應用）
│   └── app.py
│
├── data/
│   ├── tsplib/                         # 測試實例
│   └── results/                        # 實驗數據 + 圖表
│       ├── *.csv                       # 1080 次 run 的數據
│       └── figures/                    # 10+ 張圖表
│
└── tests/                              # 單元測試（4 個測試檔）
    ├── test_crossover.py
    ├── test_lineage.py
    ├── test_fitness.py
    └── test_ga_engine.py
```
