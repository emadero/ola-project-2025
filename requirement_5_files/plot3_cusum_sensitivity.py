import numpy as np
import matplotlib.pyplot as plt
import os
from requirement_5_files.sliding_window_cusum import SlidingWindowCUCB
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment

# Config
fixed_delta = 0.03
thresholds = np.linspace(0.8, 2.0, 10)
prices = [round(p, 1) for p in [0.1 * i for i in range(1, 11)]]
n_products = 8
total_rounds = 1000
production_capacity = 500 * n_products
window_size = 20
n_intervals = 8

regrets = []
resets = []

for thres in thresholds:
    env = MultiProductPiecewiseStationaryEnvironment(
        n_products=n_products,
        prices=prices,
        production_capacity=production_capacity,
        total_rounds=total_rounds,
        n_intervals=n_intervals,
        random_seed=42
    )
    agent = SlidingWindowCUCB(n_products=n_products, prices=prices, window_size=window_size,
                              delta=fixed_delta, cusum_threshold=thres)
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

    regrets.append(agent.get_cumulative_regret())
    resets.append(sum(agent.get_reset_counts()))
    print(f"Threshold={thres:.2f} → Regret={regrets[-1]:.1f}, Resets={resets[-1]}")

# Plot
os.makedirs("results/data/cusum_sensitivity", exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(thresholds, regrets, marker='o')
plt.title(f"Sensitivity Analysis: Regret vs Threshold (delta={fixed_delta})")
plt.xlabel("CUSUM Threshold")
plt.ylabel("Cumulative Regret")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/cusum_sensitivity/regret_vs_threshold.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(thresholds, resets, marker='o', color='orange')
plt.title(f"Sensitivity Analysis: Reset Count vs Threshold (delta={fixed_delta})")
plt.xlabel("CUSUM Threshold")
plt.ylabel("Reset Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/cusum_sensitivity/resets_vs_threshold.png")
plt.show()
