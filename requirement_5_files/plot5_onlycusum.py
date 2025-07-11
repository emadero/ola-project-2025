import numpy as np
import matplotlib.pyplot as plt
import os
from sliding_window_cusum import SlidingWindowCUCB
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment

# Parameters
prices = [round(p, 1) for p in [0.1 * i for i in range(1, 11)]]
n_products = 8
total_rounds = 1000
production_capacity = 500 * n_products
n_intervals_list = [2, 4, 8, 16]
window_sizes = [5, 10, 20, 40]
delta = 0.0217
threshold = 1.58

# Store results
cusum_regret_total = []
cusum_revenue_total = []
pd_regret = []
pd_revenue = []

for ws in window_sizes:
    regrets_cusum = []
    regrets_pd = []
    revenues_cusum = []
    revenues_pd = []

    for ni in n_intervals_list:
        config = dict(
            n_products=n_products,
            prices=prices,
            production_capacity=production_capacity,
            total_rounds=total_rounds,
            n_intervals=ni,
            random_seed=42
        )

        # --- SW-CUCB Agent (With CUSUM) ---
        env2 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent2 = SlidingWindowCUCB(n_products=n_products, prices=prices,
                                   window_size=ws, delta=delta, cusum_threshold=threshold)
        env2.reset()
        revenue2 = 0
        while True:
            sel = agent2.select_prices()
            sel = {p: sel[p] for p in range(len(sel))} if isinstance(sel, list) else sel
            _, rew, done = env2.step(sel)
            rew = {p: rew[p] for p in range(len(rew))} if isinstance(rew, list) else rew
            valuations = env2.get_buyer_valuations()
            agent2.update(sel, rew)
            agent2.track_regret(rew, valuations)
            revenue2 += sum(rew.values())
            if done:
                break
        regrets_cusum.append(agent2.get_cumulative_regret())
        revenues_cusum.append(revenue2)

        # --- PrimalDual Agent ---
        env3 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent3 = PrimalDualMultipleProducts(
            price_candidates=prices,
            n_products=n_products,
            inventory=production_capacity,
            n_rounds=total_rounds,
            learning_rate=0.01
        )
        history = agent3.run(env3)
        pd_regret_val = 0.0  # Placeholder for regret
        pd_revenue_val = sum(history["revenues"])
        regrets_pd.append(pd_regret_val)
        revenues_pd.append(pd_revenue_val)

    cusum_regret_total.append(regrets_cusum)
    pd_regret.append(regrets_pd)
    cusum_revenue_total.append(revenues_cusum)
    pd_revenue.append(revenues_pd)

# Plot output
os.makedirs("results/data/sensitivity_comparison", exist_ok=True)

# Regret Plot
plt.figure(figsize=(10, 5))
for i, ws in enumerate(window_sizes):
    plt.plot(n_intervals_list, cusum_regret_total[i], label=f"CUSUM SW-CUCB (ws={ws})", linestyle="-")
plt.title("Regret Sensitivity: CUSUM SW-CUCB vs PrimalDual")
plt.xlabel("n_intervals")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/sensitivity_comparison/regret_comparison_with_cusum.png")
plt.show()

# Revenue Plot
plt.figure(figsize=(10, 5))
for i, ws in enumerate(window_sizes):
    plt.plot(n_intervals_list, cusum_revenue_total[i], label=f"CUSUM SW-CUCB (ws={ws})", linestyle="-")
plt.plot(n_intervals_list, pd_revenue[0], label="PrimalDual", linestyle="dashdot", color="black")
plt.title("Cumulative Revenue: CUSUM SW-CUCB vs PrimalDual")
plt.xlabel("n_intervals")
plt.ylabel("Cumulative Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/sensitivity_comparison/revenue_comparison_with_cusum.png")
plt.show()
