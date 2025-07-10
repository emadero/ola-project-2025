import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sliding_window_cusum import SlidingWindowCUCB
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment

# Parameters
window_sizes = [5, 10, 20, 40]
intervals = [2, 4, 8, 16]
prices = [round(p, 1) for p in [0.1 * i for i in range(1, 11)]]
n_products = 8
total_rounds = 1000
production_capacity = 500 * n_products
delta = 0.03
threshold = 1.02

# Store results
regret_cusum = []
regret_standard = []
revenue_cusum = []
revenue_standard = []

for ws in window_sizes:
    row_cusum = {}
    row_std = {}
    row_rev_cusum = {}
    row_rev_std = {}
    for ni in intervals:
        config = dict(
            n_products=n_products,
            prices=prices,
            production_capacity=production_capacity,
            total_rounds=total_rounds,
            n_intervals=ni,
            random_seed=42
        )

        # With CUSUM
        env1 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent1 = SlidingWindowCUCB(n_products=n_products, prices=prices, window_size=ws,
                                   delta=delta, cusum_threshold=threshold)
        env1.reset()
        reward_sum = 0
        while True:
            sel = agent1.select_prices()
            _, rew, done = env1.step(sel)
            rew = {p: rew[p] for p in range(len(rew))}
            valuations = env1.get_buyer_valuations()
            agent1.update(sel, rew)
            agent1.track_regret(rew, valuations)
            reward_sum += sum(rew.values())
            if done:
                break
        row_cusum[ni] = agent1.get_cumulative_regret()
        row_rev_cusum[ni] = reward_sum

        # Without CUSUM
        env2 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent2 = SlidingWindowCUCB(n_products=n_products, prices=prices, window_size=ws,
                                   delta=None, cusum_threshold=None)
        env2.reset()
        reward_sum = 0
        while True:
            sel = agent2.select_prices()
            _, rew, done = env2.step(sel)
            rew = {p: rew[p] for p in range(len(rew))}
            valuations = env2.get_buyer_valuations()
            agent2.update(sel, rew)
            agent2.track_regret(rew, valuations)
            reward_sum += sum(rew.values())
            if done:
                break
        row_std[ni] = agent2.get_cumulative_regret()
        row_rev_std[ni] = reward_sum

        print(f"✅ ws={ws}, intervals={ni} | No CUSUM: {row_std[ni]:.1f} | CUSUM: {row_cusum[ni]:.1f}")

    regret_cusum.append(row_cusum)
    regret_standard.append(row_std)
    revenue_cusum.append(row_rev_cusum)
    revenue_standard.append(row_rev_std)

# Convert to DataFrames
df_cusum = pd.DataFrame(regret_cusum, index=window_sizes)
df_std = pd.DataFrame(regret_standard, index=window_sizes)
df_rev_cusum = pd.DataFrame(revenue_cusum, index=window_sizes)
df_rev_std = pd.DataFrame(revenue_standard, index=window_sizes)

# Save CSVs
os.makedirs("results/data/cusum_comparison", exist_ok=True)
df_std.to_csv("results/data/cusum_comparison/regret_nocusum.csv")
df_cusum.to_csv("results/data/cusum_comparison/regret_cusum.csv")
df_rev_std.to_csv("results/data/cusum_comparison/revenue_nocusum.csv")
df_rev_cusum.to_csv("results/data/cusum_comparison/revenue_cusum.csv")

# Regret heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.heatmap(df_std, annot=True, fmt=".1f", cmap="OrRd", ax=axes[0])
axes[0].set_title("SW-CUCB Regret (No CUSUM)")
axes[0].set_xlabel("n_intervals")
axes[0].set_ylabel("window_size")
sns.heatmap(df_cusum, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
axes[1].set_title("SW-CUCB Regret (With CUSUM)")
axes[1].set_xlabel("n_intervals")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig("results/data/cusum_comparison/cusum_vs_nocusum_heatmaps.png")
plt.show()

# Regret difference
diff = df_cusum - df_std
plt.figure(figsize=(6, 4))
sns.heatmap(diff, annot=True, fmt=".1f", center=0, cmap="coolwarm")
plt.title("CUSUM - No CUSUM (Regret Difference)")
plt.xlabel("n_intervals")
plt.ylabel("window_size")
plt.tight_layout()
plt.savefig("results/data/cusum_comparison/difference_heatmap.png")
plt.show()

# Revenue heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.heatmap(df_rev_std, annot=True, fmt=".0f", cmap="OrRd", ax=axes[0])
axes[0].set_title("SW-CUCB Revenue (No CUSUM)")
axes[0].set_xlabel("n_intervals")
axes[0].set_ylabel("window_size")
sns.heatmap(df_rev_cusum, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[1])
axes[1].set_title("SW-CUCB Revenue (With CUSUM)")
axes[1].set_xlabel("n_intervals")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig("results/data/cusum_comparison/revenue_heatmaps.png")
plt.show()

# Revenue difference
diff_rev = df_rev_cusum - df_rev_std
plt.figure(figsize=(6, 4))
sns.heatmap(diff_rev, annot=True, fmt=".0f", center=0, cmap="coolwarm")
plt.title("CUSUM - No CUSUM (Revenue Difference)")
plt.xlabel("n_intervals")
plt.ylabel("window_size")
plt.tight_layout()
plt.savefig("results/data/cusum_comparison/revenue_diff_heatmap.png")
plt.show()
