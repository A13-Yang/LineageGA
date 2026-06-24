# Phase 1：基礎建設 — 專案架構 + TSPLIB + GA 基本元件

**時程：第 1-2 週**
**前置依賴：無（起始 Phase）**

---

## 目標

建立專案骨架，完成所有 GA 底層元件，使得一個「無血統追蹤的標準 GA」能在 TSP 上正確運行並收斂。

> 這是所有後續工作的地基。Phase 1 結束時，應能跑一次簡單的 `eil51` GA 並看到適應度隨世代下降。

---

## 任務清單

### Week 1：專案骨架 + 資料層

- [ ] **T1.1** 建立專案目錄結構（`src/`、`experiments/`、`analysis/`、`data/`、`tests/`）
- [ ] **T1.2** 撰寫 `pyproject.toml`，定義所有依賴（numpy、scipy、matplotlib、seaborn、plotly、pandas、tqdm、joblib、dash）
- [ ] **T1.3** 建立虛擬環境並安裝依賴，確認 `import` 全部正常
- [ ] **T1.4** 實作 `src/tsplib_parser.py`
  - 解析 `.tsp` 檔案（EUC_2D 格式）
  - 解析 `.opt.tour` 最優解檔案
  - 產生距離矩陣 `np.ndarray`
- [ ] **T1.5** 下載 TSPLIB 測試實例至 `data/tsplib/`
  - `eil51.tsp` + `eil51.opt.tour`
  - `kroA100.tsp` + `kroA100.opt.tour`
- [ ] **T1.6** 實作 `src/individual.py`
  - `Individual` dataclass（id, genes, fitness, generation, parent_ids, ancestry, lc）
  - 全域 ID 產生器
- [ ] **T1.7** 實作 `src/fitness.py`
  - `build_distance_matrix(cities)` → 預計算距離矩陣
  - `calculate_tour_length(genes, dist_matrix)` → 路徑總長度

### Week 2：GA 運算子 + 族群管理

- [ ] **T1.8** 實作 `src/crossover.py`
  - Order Crossover (OX)
  - Partially Mapped Crossover (PMX)
  - 每次交叉後 assert 排列合法性（無重複、無遺漏）
- [ ] **T1.9** 實作 `src/mutation.py`
  - Swap Mutation（以機率 p 交換兩個城市）
- [ ] **T1.10** 實作 `src/selection.py`
  - Tournament Selection（k=3），先只做標準版
  - 統一介面：`select_parents(population, strategy, **kwargs)`
- [ ] **T1.11** 實作 `src/population.py`
  - `create_initial_population(n_cities, pop_size)` → 隨機排列
  - `elitism(population, n_elite)` → 保留 top N
  - `calculate_diversity(population)` → edge-difference 多樣性
- [ ] **T1.12** 撰寫基礎單元測試
  - `tests/test_crossover.py`：OX/PMX 輸出為合法排列（100 次隨機測試）
  - `tests/test_fitness.py`：手動構造已知路徑，驗證長度計算正確

---

## 交付物

| 檔案 | 狀態 | 驗證方式 |
|---|---|---|
| `pyproject.toml` | 新建 | `pip install -e .` 成功 |
| `src/tsplib_parser.py` | 新建 | 能正確解析 eil51.tsp，距離矩陣維度 = 51×51 |
| `src/individual.py` | 新建 | Individual 物件可正確建立 |
| `src/fitness.py` | 新建 | eil51 最優解路徑長度 = 426 |
| `src/crossover.py` | 新建 | 100 次隨機 OX/PMX 全部產生合法排列 |
| `src/mutation.py` | 新建 | Swap 後排列仍合法 |
| `src/selection.py` | 新建 | Tournament 選出的個體確實來自族群 |
| `src/population.py` | 新建 | 初始族群大小正確、無重複個體 |
| `tests/test_crossover.py` | 新建 | `pytest` 全部通過 |
| `tests/test_fitness.py` | 新建 | `pytest` 全部通過 |

---

## Phase 1 完成標準（Gate）

> **必須全部滿足才進入 Phase 2：**

- [x] 所有 `src/` 模組可獨立 import 不報錯
- [x] `eil51.tsp` 能被正確解析為 51 個城市座標
- [x] OX/PMX 交叉 100 次測試零失敗
- [x] 手動拼接一個簡單的 GA 迴圈（不含血統追蹤），在 eil51 上跑 100 代，觀察到適應度下降趨勢

---

## 風險與注意事項

| 風險 | 對策 |
|---|---|
| PMX 映射衝突處理不正確 | 用小規模（5 城市）手動驗算，確認映射鏈正確解析 |
| TSPLIB 格式變體（非 EUC_2D） | Phase 1 只支援 EUC_2D，後續按需擴充 |
| 距離矩陣記憶體（pr1002 = 1002×1002） | 約 8 MB，不構成問題 |
