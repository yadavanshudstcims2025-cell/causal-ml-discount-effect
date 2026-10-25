# ============================================================
# STEP 7: ROBUSTNESS ANALYSIS ACROSS RANDOM SEEDS
# ============================================================
# Repeats the data generation of Step 1 for ten random seeds and
# compares the naive ATE with the true ATE each time.
# ============================================================

import os
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)


def naive_ate(seed, n=10_000):
    np.random.seed(seed)
    pp  = np.random.randint(0, 50, n)           # previous_purchases
    aov = np.random.uniform(10, 500, n)         # avg_order_value
    dsl = np.random.randint(1, 365, n)          # days_since_last_purchase (drawn to keep the random stream identical to Step 1)
    em  = np.random.binomial(1, 0.35, n)        # email_clicked
    pt  = np.clip(0.20 + 0.50 * aov / 500, 0.05, 0.90)
    T   = (np.random.random(n) < pt).astype(int)
    b   = np.clip(0.10 + 0.002 * pp + 0.00015 * aov + 0.05 * em, 0.01, 0.80)
    raw = 0.005 + 0.00015 * pp + 0.000015 * aov + 0.002 * em
    eff = raw * 0.015 / raw.mean()
    p1  = np.clip(b + eff, 0.01, 0.95)
    Y   = (np.random.random(n) < np.where(T == 1, p1, b)).astype(int)
    return Y[T == 1].mean() - Y[T == 0].mean(), eff.mean()


seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
res = pd.DataFrame([naive_ate(s) for s in seeds], columns=["naive", "true"], index=seeds)
res.index.name = "seed"
res["bias"] = res["naive"] - res["true"]

print("=" * 65)
print("STEP 7: ROBUSTNESS ANALYSIS")
print("=" * 65)
print(res.round(4))
print("-" * 65)
print(f"Mean naive ATE:         {res['naive'].mean():.5f}")
print(f"SD of naive ATE:        {res['naive'].std():.5f}")
print(f"Mean bias:              {res['bias'].mean():.5f}")
print(f"Mean absolute error:    {res['bias'].abs().mean():.5f}")
print(f"Maximum absolute error: {res['bias'].abs().max():.5f}")

res.round(5).to_csv("results/robustness_results.csv")
