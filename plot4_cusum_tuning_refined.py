import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product

from sliding_window_cusum import SlidingWindowCUCB
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment

# Parameters
deltas = np.linspace(0.005, 0.03, 10)
thresholds = np.linspace(0.8, 1.8, 10)
prices = [round(p, 1) for p in [0.1 * i for i in range(1, 11)]]
n_products = 8
total_rounds = 1000
production_capacity = 500 * n_products
window_size = 20
n_intervals = 8

os.makedirs("results/data/cusum_tuning_refined", exist_ok=True)

records = []

for delta, thres in product(deltas, thresholds):
    config = dict(
        n_products=n_products,
        prices=prices,
        production_capacity=production_capacity,
        total_rounds=total_rounds,
        n_intervals=n_intervals,
        random_seed=42
    )
    env = MultiProductPiecewiseStationaryEnvironment(**config)
    agent = SlidingWindowCUCB(n_products=n_products, prices=prices, window_size=window_size,
                              delta=delta, cusum_threshold=thres)
    env.reset()

    while True:
        sel = agent.select_prices()
        sel = {p: sel[p] for p in range(len(sel))} if isinstance(sel, list) else sel
        _, rew, done = env.step(sel)
        rew = {p: rew[p] for p in range(len(rew))} if isinstance(rew, list) else rew
        valuations = env.get_buyer_valuations()
        agent.update(sel, rew)
        agent.track_regret(rew, valuations)
        if done:
            break

    regret = agent.get_cumulative_regret()
    resets = sum(agent.get_reset_counts())
    records.append({
        "delta": round(delta, 4),
        "threshold": round(thres, 2),
        "regret": regret,
        "resets": resets
    })
    print(f"✅ δ={delta:.4f}, τ={thres:.2f} → Regret={regret:.1f}, Resets={resets}")

df = pd.DataFrame(records)
df.to_csv("results/data/cusum_tuning_refined/grid_results.csv", index=False)

# Plot heatmap of regret
pivot = df.pivot(index="threshold", columns="delta", values="regret")
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("CUSUM Tuning: Regret")
plt.xlabel("delta")
plt.ylabel("threshold")
plt.tight_layout()
plt.savefig("results/data/cusum_tuning_refined/cusum_regret_heatmap.png")
plt.show()

# Plot heatmap of reset counts
pivot_reset = df.pivot(index="threshold", columns="delta", values="resets")
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_reset, annot=True, fmt=".0f", cmap="rocket_r")
plt.title("CUSUM Tuning : Reset Count")
plt.xlabel("delta")
plt.ylabel("threshold")
plt.tight_layout()
plt.savefig("results/data/cusum_tuning_refined/cusum_reset_heatmap.png")
plt.show()
