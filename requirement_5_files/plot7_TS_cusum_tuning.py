import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment
from thompson_sampling_cusum import ThompsonSamplingCUSUM

# Parameters
n_products = 8
prices = [round(p, 1) for p in np.linspace(0.1, 1.0, 10)]
total_rounds = 1000
production_capacity = 500 * n_products
n_intervals = 8

delta_values = np.linspace(0.005, 0.05, 10)
threshold_values = np.linspace(0.5, 3.0, 10)

results = []
best_revenue = -np.inf
best_params = {}

# Grid Search
for delta in delta_values:
    for threshold in threshold_values:
        env = MultiProductPiecewiseStationaryEnvironment(
            n_products=n_products,
            prices=prices,
            production_capacity=production_capacity,
            total_rounds=total_rounds,
            n_intervals=n_intervals,
            random_seed=42
        )

        agent = ThompsonSamplingCUSUM(
            n_products=n_products,
            prices=prices,
            delta=delta,
            cusum_threshold=threshold
        )

        revenue = 0
        done = False

        env.reset()
        while not done:
            action = agent.act()
            _, reward, done = env.step(action)
            valuations = env.get_buyer_valuations()
            agent.update(action, reward)
            agent.track_regret(reward, valuations)
            revenue += sum(reward.values())

        regret = agent.get_cumulative_regret()
        resets = sum(len(r) == 0 for r in agent.recent_rewards)

        results.append({
            "delta": delta,
            "threshold": threshold,
            "revenue": revenue,
            "resets": resets,
            "regret": regret
        })

        if revenue > best_revenue:
            best_revenue = revenue
            best_params = {"delta": delta, "threshold": threshold}

# Save Results
df = pd.DataFrame(results)
os.makedirs("results/data/tuning", exist_ok=True)
df.to_csv("results/data/tuning/ts_cusum_tuning_final.csv", index=False)

# Round axis labels for readability
delta_labels = np.round(delta_values, 3)
threshold_labels = np.round(threshold_values, 2)

# Revenue Heatmap
heatmap_data = df.pivot(index="delta", columns="threshold", values="revenue")
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu",
            xticklabels=threshold_labels, yticklabels=delta_labels)
plt.title("TS-CUSUM Revenue Heatmap")
plt.xlabel("Threshold")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig("results/data/tuning/ts_cusum_revenue_heatmap_final.png")
plt.show()

# Reset Count Heatmap
reset_data = df.pivot(index="delta", columns="threshold", values="resets")
plt.figure(figsize=(10, 6))
sns.heatmap(reset_data, annot=True, fmt=".0f", cmap="OrRd",
            xticklabels=threshold_labels, yticklabels=delta_labels)
plt.title("TS-CUSUM Reset Count Heatmap")
plt.xlabel("Threshold")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig("results/data/tuning/ts_cusum_reset_heatmap_final.png")
plt.show()

# Regret Heatmap
regret_data = df.pivot(index="delta", columns="threshold", values="regret")
plt.figure(figsize=(10, 6))
sns.heatmap(regret_data, annot=True, fmt=".0f", cmap="YlOrBr",
            xticklabels=threshold_labels, yticklabels=delta_labels)
plt.title("TS-CUSUM Regret Heatmap")
plt.xlabel("Threshold")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig("results/data/tuning/ts_cusum_regret_heatmap_final.png")
plt.show()

# Print Best
print("\\n🏆 Best Parameters by Revenue:")
print(f"   ➤ delta     = {best_params['delta']}")
print(f"   ➤ threshold = {best_params['threshold']}")
print(f"   ➤ revenue   = {best_revenue}")
