"""
快速分析腳本：四策略 x 5 runs x eil51 (300 代)
在專案根目錄下執行：
    conda activate bloodline-ga
    python run_analysis.py
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
from experiments.config import config_a, config_b, config_c, config_d
from experiments.runner import run_single
from src.lineage import LineageTracker

# ── 實驗設定 ──
N_RUNS = 5
N_GEN = 300
OPTIMAL = 426.0

strategies = {
    "tournament": config_a,
    "elite":      config_b,
    "poor":       config_c,
    "random":     config_d,
}

# ── 跑實驗 ──
print("開始跑實驗 (4 策略 × 5 runs × 300 代 on eil51)...")
results = {}
for name, factory in strategies.items():
    cfg = factory("eil51", n_generations=N_GEN, snapshot_interval=50)
    runs = []
    for seed in range(N_RUNS):
        r = run_single(cfg, seed, write_artifacts=False)
        runs.append(r)
        print(f"  {name} seed={seed} done: best={r.final_best_individual.fitness:.1f}")
    results[name] = runs

# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  四策略 eil51 實驗結果摘要")
print("=" * 80)

for name, runs in results.items():
    final_bests = [r.final_best_individual.fitness for r in runs]
    final_lcs   = [r.history[-1].avg_lc for r in runs]
    final_divs  = [r.history[-1].diversity for r in runs]
    gaps = [(f - OPTIMAL) / OPTIMAL * 100 for f in final_bests]

    all_corrs = [
        rec.lc_fitness_correlation
        for r in runs for rec in r.history
        if rec.lc_fitness_correlation != 0.0
    ]
    upset_rates = [r.history[-1].upset_offspring_rate for r in runs]

    print(f"\n--- {name.upper()} ---")
    print(f"  最佳適應度:       {np.mean(final_bests):.1f} ± {np.std(final_bests):.1f}")
    print(f"  與最優解差距:     {np.mean(gaps):.1f}% ± {np.std(gaps):.1f}%")
    print(f"  最終平均 LC:      {np.mean(final_lcs):.4f} ± {np.std(final_lcs):.4f}")
    print(f"  最終族群多樣性:   {np.mean(final_divs):.4f} ± {np.std(final_divs):.4f}")
    print(f"  LC-Fitness 相關 (跨世代均): {np.mean(all_corrs):+.4f}")
    print(f"  最終代逆勢子代率: {np.mean(upset_rates):.4f}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  LC 趨勢隨世代變化 (每 50 代)")
print("=" * 80)

checkpoints = [0, 50, 100, 150, 200, 250, 300]
for name, runs in results.items():
    print(f"\n--- {name.upper()} ---")
    print(f"  {'Gen':>5}  {'LC':>8}  {'LC-Fit相關':>11}  {'逆勢率':>8}  {'多樣性':>8}  {'最佳適應度':>10}")
    for gen in checkpoints:
        lcs    = [r.history[gen].avg_lc for r in runs]
        corrs  = [r.history[gen].lc_fitness_correlation for r in runs]
        upsets = [r.history[gen].upset_offspring_rate for r in runs]
        divs   = [r.history[gen].diversity for r in runs]
        bests  = [r.history[gen].best_fitness for r in runs]
        print(f"  {gen:5d}  {np.mean(lcs):8.4f}  {np.mean(corrs):+11.4f}  {np.mean(upsets):8.4f}  {np.mean(divs):8.4f}  {np.mean(bests):10.1f}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  精英 vs 劣質育種：逆勢子代率比較")
print("=" * 80)

for name in ["elite", "poor", "tournament", "random"]:
    runs = results[name]
    all_upsets = [rec.upset_offspring_rate for r in runs for rec in r.history[1:]]
    print(f"  {name.upper():>10}: 整體平均逆勢子代率 = {np.mean(all_upsets):.4f} (±{np.std(all_upsets):.4f})")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  有效始祖數 & 血統多樣性（最終代快照）")
print("=" * 80)

for name, runs in results.items():
    tracker = LineageTracker(runs[0].founder_qualities)
    entropies = []
    eff_founders = []
    ancestry_sizes = []
    for r in runs:
        last_gen = r.history[-1]
        if last_gen.individuals:
            for ind in last_gen.individuals:
                entropies.append(tracker.compute_ancestry_entropy(ind.ancestry))
                eff_founders.append(tracker.compute_effective_founders(ind.ancestry))
                ancestry_sizes.append(len(ind.ancestry))
    if entropies:
        print(f"  {name.upper():>10}: 有效始祖數={np.mean(eff_founders):6.2f}±{np.std(eff_founders):.2f}  "
              f"血統熵={np.mean(entropies):.3f}±{np.std(entropies):.3f}  "
              f"AP字典大小={np.mean(ancestry_sizes):.1f}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  ★ 意外發現分析 ★")
print("=" * 80)

# 1. 劣質育種的逆勢子代
poor_upset_early = []
poor_upset_late = []
for r in results["poor"]:
    for rec in r.history[1:51]:
        poor_upset_early.append(rec.upset_offspring_rate)
    for rec in r.history[251:]:
        poor_upset_late.append(rec.upset_offspring_rate)

elite_upset_early = []
elite_upset_late = []
for r in results["elite"]:
    for rec in r.history[1:51]:
        elite_upset_early.append(rec.upset_offspring_rate)
    for rec in r.history[251:]:
        elite_upset_late.append(rec.upset_offspring_rate)

print(f"\n  [發現 1] 劣質育種逆勢子代率：")
print(f"    前 50 代: {np.mean(poor_upset_early):.4f}")
print(f"    後 50 代: {np.mean(poor_upset_late):.4f}")
print(f"  精英育種逆勢子代率：")
print(f"    前 50 代: {np.mean(elite_upset_early):.4f}")
print(f"    後 50 代: {np.mean(elite_upset_late):.4f}")

# 2. LC-fitness 相關性方向
print(f"\n  [發現 2] LC-Fitness 相關性方向（TSP 是最小化問題）：")
for name, runs in results.items():
    corrs = [rec.lc_fitness_correlation for r in runs for rec in r.history[50:]]
    sign = "正相關 (LC高→fitness高)" if np.mean(corrs) > 0 else "負相關 (LC高→fitness低/路徑短→好)"
    print(f"    {name.upper():>10}: r = {np.mean(corrs):+.4f}  → {sign}")

# 3. 收斂速度比較
print(f"\n  [發現 3] 收斂速度（達到最優解 +15% 的世代數）：")
for name, runs in results.items():
    target = OPTIMAL * 1.15
    conv_gens = []
    for r in runs:
        for rec in r.history:
            if rec.best_fitness <= target:
                conv_gens.append(rec.generation)
                break
        else:
            conv_gens.append(N_GEN)  # 未收斂
    print(f"    {name.upper():>10}: 平均 {np.mean(conv_gens):.0f} 代 (range: {min(conv_gens)}-{max(conv_gens)})")

# 4. 最終精英組多樣性 vs 隨機組
print(f"\n  [發現 4] 最終代族群多樣性比較：")
for name in ["elite", "tournament", "random", "poor"]:
    divs = [r.history[-1].diversity for r in results[name]]
    print(f"    {name.upper():>10}: {np.mean(divs):.4f}")

# 5. 劣質育種是否有產生比精英育種更好的個體？
print(f"\n  [發現 5] 各策略歷史最佳解：")
for name, runs in results.items():
    all_time_bests = [min(rec.best_fitness for rec in r.history) for r in runs]
    print(f"    {name.upper():>10}: best of bests = {min(all_time_bests):.1f}  (gap={((min(all_time_bests)-OPTIMAL)/OPTIMAL*100):.1f}%)")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
