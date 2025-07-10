import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from requirement_5_files.sliding_window_cusum import SlidingWindowCUCB
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment

def run_final_comparison(n_products=8, capacity_multiplier=500, use_cusum=True, track_regret=True):
    total_rounds = 1000
    prices = [round(p, 1) for p in [0.1 * i for i in range(1, 11)]]
    production_capacity = capacity_multiplier * n_products
    n_intervals = 8
    window_size = 20

    env_sw = MultiProductPiecewiseStationaryEnvironment(
        n_products=n_products,
        prices=prices,
        production_capacity=production_capacity,
        total_rounds=total_rounds,
        n_intervals=n_intervals,
        random_seed=42
    )
    if use_cusum:
        agent_sw = SlidingWindowCUCB(
            n_products=n_products,
            prices=prices,
            window_size=window_size,
            delta=0.03,
            cusum_threshold=1.0
        )
    else:
        agent_sw = SlidingWindowCUCB(
            n_products=n_products,
            prices=prices,
            window_size=window_size,
            delta=None,
            cusum_threshold=None
        )
    env_sw.reset()
    revenues_sw = []
    regrets_sw = []

    while True:
        sel = agent_sw.select_prices()
        sel = {p: sel[p] for p in range(len(sel))}
        _, rew, done = env_sw.step(sel)
        rew = {p: rew[p] for p in range(len(rew))}
        revenues_sw.append(sum(rew.values()))
        if track_regret:
            valuations = env_sw.get_buyer_valuations()
            agent_sw.track_regret(rew, valuations)
            regrets_sw.append(agent_sw.get_cumulative_regret())
        agent_sw.update(sel, rew)
        if done:
            break

    env_pd = MultiProductPiecewiseStationaryEnvironment(
        n_products=n_products,
        prices=prices,
        production_capacity=production_capacity,
        total_rounds=total_rounds,
        n_intervals=n_intervals,
        random_seed=42
    )
    agent_pd = PrimalDualMultipleProducts(
        price_candidates=prices,
        n_products=n_products,
        inventory=production_capacity,
        n_rounds=total_rounds,
        learning_rate=0.01
    )
    history_pd = agent_pd.run(env_pd)

    os.makedirs("results/figures/final_comparison", exist_ok=True)
    os.makedirs("results/data/final_comparison", exist_ok=True)

    min_len = min(len(revenues_sw), len(history_pd["revenues"]))
    revenues_sw = revenues_sw[:min_len]
    revenues_pd = history_pd["revenues"][:min_len]
    rounds = np.arange(1, min_len + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(np.cumsum(revenues_sw), label="SW-CUCB", color="blue")
    ax1.plot(np.cumsum(revenues_pd), label="PrimalDual", linestyle="--", color="black")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Revenue")
    ax1.set_title(f"Final Comparison (n={n_products}, capacity={capacity_multiplier}×)")
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    fig_name = f"results/figures/final_comparison/revenue_n{n_products}_cap{capacity_multiplier}_cusum{use_cusum}_regret{track_regret}.png"
    plt.savefig(fig_name)
    plt.close()

    data = {
        "round": rounds,
        "revenue_sw": revenues_sw,
        "revenue_pd": revenues_pd
    }

    if track_regret:
        data["regret_sw"] = regrets_sw[:min_len]

    df = pd.DataFrame(data)
    csv_name = f"results/data/final_comparison/data_n{n_products}_cap{capacity_multiplier}_cusum{use_cusum}_regret{track_regret}.csv"
    df.to_csv(csv_name, index=False)

# Example usage
if __name__ == "__main__":
    try:
        n = int(input("🔢 How many products? (e.g., 3 or 8): "))
    except ValueError:
        print("⚠️ Invalid input. Using default n_products = 8.")
        n = 8

    run_final_comparison(
        n_products=n,
        capacity_multiplier=500,
        use_cusum=True,
        track_regret=True
    )
