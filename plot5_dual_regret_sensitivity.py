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

# Store results
sw_cucb_regret = []
sw_cucb_revenue = []
pd_regret = []
pd_revenue = []

for ws in window_sizes:
    regrets_sw = []
    regrets_pd = []
    revenues_sw = []
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

        # --- SW-CUCB Agent ---
        env1 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent1 = SlidingWindowCUCB(n_products=n_products, prices=prices,
                                   window_size=ws, delta=None, cusum_threshold=None)
        env1.reset()
        revenue1 = 0
        while True:
            sel = agent1.select_prices()
            sel = {p: sel[p] for p in range(len(sel))} if isinstance(sel, list) else sel
            _, rew, done = env1.step(sel)
            rew = {p: rew[p] for p in range(len(rew))} if isinstance(rew, list) else rew
            valuations = env1.get_buyer_valuations()
            agent1.update(sel, rew)
            agent1.track_regret(rew, valuations)
            revenue1 += sum(rew.values())
            if done:
                break
        regrets_sw.append(agent1.get_cumulative_regret())
        revenues_sw.append(revenue1)

        # --- PrimalDual Agent ---
        env2 = MultiProductPiecewiseStationaryEnvironment(**config)
        agent2 = PrimalDualMultipleProducts(
            price_candidates=prices,
            n_products=n_products,
            inventory=production_capacity,
            n_rounds=total_rounds,
            learning_rate=0.01
        )
        history = agent2.run(env2)
        pd_regret_val = 0.0  # you can add proper regret computation here if needed
        pd_revenue_val = sum(history["revenues"])
        revenues_pd.append(pd_revenue_val)
        regrets_pd.append(pd_regret_val)

    sw_cucb_regret.append(regrets_sw)
    pd_regret.append(regrets_pd)
    sw_cucb_revenue.append(revenues_sw)
    pd_revenue.append(revenues_pd)

# Plot output
os.makedirs("results/data/sensitivity_comparison", exist_ok=True)

# Regret Plot
plt.figure(figsize=(10, 5))
for i, ws in enumerate(window_sizes):
    plt.plot(n_intervals_list, sw_cucb_regret[i], label=f"SW-CUCB (ws={ws})")
plt.title("SW-CUCB Regret Sensitivity vs Intervals")
plt.xlabel("n_intervals")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/sensitivity_comparison/sw_cucb_regret_sensitivity.png")
plt.show()

# Revenue Plot
plt.figure(figsize=(10, 5))
for i, ws in enumerate(window_sizes):
    plt.plot(n_intervals_list, sw_cucb_revenue[i], label=f"SW-CUCB (ws={ws})")
plt.plot(n_intervals_list, pd_revenue[0], label="PrimalDual", linestyle="--", color="black")
plt.title("Cumulative Revenue: SW-CUCB vs PrimalDual")
plt.xlabel("n_intervals")
plt.ylabel("Cumulative Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/data/sensitivity_comparison/revenue_comparison.png")
plt.show()
