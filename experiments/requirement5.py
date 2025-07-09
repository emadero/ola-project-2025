from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment
from algorithms.multiple_products.sliding_window import SlidingWindowCUCB
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts
import matplotlib.pyplot as plt
import pandas as pd
import os

def align_series(y, rounds):
    return pd.Series(y, index=rounds).cumsum()

def to_price_dict(raw_sel):
    """Ensure prices_selected is a dict for env.step()."""
    if isinstance(raw_sel, list):
        return {p: raw_sel[p] for p in range(len(raw_sel))}
    return raw_sel

def to_reward_dict(raw_rewards):
    """Ensure rewards is a dict for agent.update()."""
    if isinstance(raw_rewards, list):
        return {p: raw_rewards[p] for p in range(len(raw_rewards))}
    return raw_rewards

def run_single_experiment():
    print("\n🚀 Running main comparison: Sliding Window CUCB vs Primal-Dual")
    prices = [0.2, 0.4, 0.6, 0.8]
    env_config = dict(
        n_products=3,
        prices=prices,
        production_capacity=60,
        total_rounds=40,
        n_intervals=4,
        random_seed=42
    )

    # --- Sliding Window CUCB run ---
    env_sw = MultiProductPiecewiseStationaryEnvironment(**env_config)
    agent_sw = SlidingWindowCUCB(n_products=3, prices=prices, window_size=10)
    env_sw.reset()

    sw_rewards = []
    while True:
        raw_sel = agent_sw.select_prices()
        prices_selected = to_price_dict(raw_sel)

        buyer_info, raw_rewards, done = env_sw.step(prices_selected)
        rewards = to_reward_dict(raw_rewards)

        agent_sw.update(prices_selected, rewards)
        sw_rewards.append(sum(rewards.values()))

        if done:
            break

    # --- Primal‐Dual run ---
    env_pd = MultiProductPiecewiseStationaryEnvironment(**env_config)
    agent_pd = PrimalDualMultipleProducts(
        price_candidates=prices,
        n_products=3,
        inventory=env_config["production_capacity"],
        n_rounds=env_config["total_rounds"],
        learning_rate=0.01
    )
    pd_history = agent_pd.run(env_pd)

    # Align & save
    min_len = min(len(sw_rewards), len(pd_history["revenues"]))
    rounds   = list(range(1, min_len + 1))
    df = pd.DataFrame({
        "round": rounds,
        "sliding_window": align_series(sw_rewards[:min_len], rounds),
        "primal_dual":    align_series(pd_history["revenues"][:min_len], rounds),
    })

    os.makedirs("results/data", exist_ok=True)
    df.to_csv("results/data/req5_comparison.csv", index=False)

    os.makedirs("results/figures", exist_ok=True)
    plt.figure()
    plt.plot(df["round"], df["sliding_window"], label="SlidingWindowCUCB")
    plt.plot(df["round"], df["primal_dual"],    label="PrimalDual")
    plt.title("Requirement 5: Algorithm Comparison")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/req5_algorithm_comparison.png")
    plt.show()
    print("✅ Saved main comparison plot and CSV.")

def run_grid_analysis():
    print("\n📊 Running sensitivity analysis (window size × intervals)")
    prices = [0.2, 0.4, 0.6, 0.8]
    window_sizes    = [5, 10, 20]
    interval_lengths = [2, 4, 8]

    for ws in window_sizes:
        for ni in interval_lengths:
            env_config = dict(
                n_products=3,
                prices=prices,
                production_capacity=60,
                total_rounds=40,
                n_intervals=ni,
                random_seed=42
            )

            # Sliding Window run
            env_sw = MultiProductPiecewiseStationaryEnvironment(**env_config)
            agent_sw = SlidingWindowCUCB(n_products=3, prices=prices, window_size=ws)
            env_sw.reset()

            sw_rewards = []
            while True:
                raw_sel = agent_sw.select_prices()
                prices_selected = to_price_dict(raw_sel)

                buyer_info, raw_rewards, done = env_sw.step(prices_selected)
                rewards = to_reward_dict(raw_rewards)

                agent_sw.update(prices_selected, rewards)
                sw_rewards.append(sum(rewards.values()))

                if done:
                    break

            # Primal‐Dual run
            env_pd = MultiProductPiecewiseStationaryEnvironment(**env_config)
            agent_pd = PrimalDualMultipleProducts(
                price_candidates=prices,
                n_products=3,
                inventory=env_config["production_capacity"],
                n_rounds=env_config["total_rounds"],
                learning_rate=0.01
            )
            pd_history = agent_pd.run(env_pd)

            # Align & save
            min_len = min(len(sw_rewards), len(pd_history["revenues"]))
            rounds   = list(range(1, min_len + 1))
            df = pd.DataFrame({
                "round": rounds,
                "sliding_window": align_series(sw_rewards[:min_len], rounds),
                "primal_dual":    align_series(pd_history["revenues"][:min_len], rounds),
            })

            fname = f"req5_ws{ws}_int{ni}"
            df.to_csv(f"results/data/{fname}.csv", index=False)

            plt.figure()
            plt.plot(df["round"], df["sliding_window"], label=f"SW-CUCB (ws={ws})")
            plt.plot(df["round"], df["primal_dual"],    label="PrimalDual")
            plt.title(f"Grid: Window={ws}, Intervals={ni}")
            plt.xlabel("Round")
            plt.ylabel("Cumulative Revenue")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/figures/{fname}.png")
            plt.close()

            print(f"✅ Saved: results/data/{fname}.csv")
            print(f"📊 Plot: results/figures/{fname}.png")

def main():
    print("📋 Requirement 5: Slightly Non-Stationary Environment Analysis")
    run_single_experiment()
    run_grid_analysis()
    print("\n🏁 All experiments completed.")

if __name__ == "__main__":
    main()
