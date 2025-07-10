# experiments/requirement4.py

"""
Requirement 4: Best-of-Both-Worlds with Multiple Products
Assigned to: Maxence Guyot

This experiment evaluates the Primal-Dual algorithm with inventory constraint
on two types of multi-product environments:
1. A stationary stochastic environment
2. A highly non-stationary environment with fast-changing valuations

The goal is to test the algorithm's adaptability in both settings.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make project modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.multi_stochastic import MultiProductStochasticEnvironment
from environments.multi_non_stationary import MultiProductHighlyNonStationaryEnvironment
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts

# Shared experiment configuration
N_PRODUCTS = 10
PRICES = [0.2, 0.3, 0.4, 0.5, 0.6]
INVENTORY = 500
ROUNDS = 90

def run_experiment(env, algo_name, title):
    """
    Run the Primal-Dual algorithm in a given environment and log results.

    Args:
        env: The environment instance (stochastic or non-stationary)
        algo_name: Label for the algorithm used (for plot legends)
        title: Title to display in the console/log
    """
    env.reset()
    algo = PrimalDualMultipleProducts(
        n_products=N_PRODUCTS,
        price_candidates=PRICES,
        inventory=INVENTORY,
        n_rounds=ROUNDS      
    )

    rewards_log = []
    cumulative_inventory = []
    total_revenue = 0
    inventory_used = 0
    rounds = []

    print(f"\n📊 {title}")
    print("=" * 60)
    
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_name = algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_log.csv"
    csv_path = os.path.join(LOG_DIR, csv_name)

    if os.path.exists(csv_path):
        os.remove(csv_path)

    log_rows = []

    for t in range(ROUNDS):
        prices, indices = algo.select_prices()
        buyer_info, rewards, done = env.step(prices)
        algo.update(indices, buyer_info["purchases"], rewards)

        round_revenue = sum(rewards.values())
        total_revenue += round_revenue
        inventory_used += sum(buyer_info["purchases"].values())

        
        log_rows.append({
            "Round": t + 1,
            "Selected Prices": prices,
            "Buyer Valuations": buyer_info["valuations"],
            "Purchases": buyer_info["purchases"],
            "Rewards": rewards,
            "Revenue": round_revenue,
            "Remaining Inventory": env.remaining_inventory
        })

        rewards_log.append(round_revenue)
        cumulative_inventory.append(inventory_used)
        rounds.append(t + 1)

        if done:
            print("\n🛑 Environment finished.")
            break

    print(f"\n💰 Total Revenue: {total_revenue:.2f}")

    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(csv_path, index=False)
    print(f"📁 Log saved to: {csv_path}")

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(rounds, cumulative_inventory, label="Inventory Used")
    axs[0].axhline(y=INVENTORY, color='r', linestyle='--', label="Inventory Limit")
    axs[0].set_ylabel("Units")
    axs[0].set_title(f"Inventory Usage - {algo_name}")
    axs[0].legend()

    axs[1].plot(rounds, rewards_log, label="Revenue per Round")
    axs[1].set_xlabel("Round")
    axs[1].set_ylabel("Revenue")
    axs[1].set_title(f"Revenue per Round - {algo_name}")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
        # --- Oracle estimation: best fixed prices ---
    def estimate_best_fixed_prices(n_simulations=1000):
        expected_rewards = np.zeros((N_PRODUCTS, len(PRICES)))
        for _ in range(n_simulations):
            buyer = env._generate_buyer()
            for pid in range(N_PRODUCTS):
                for i, price in enumerate(PRICES):
                    expected_rewards[pid, i] += float(buyer.valuations[pid] >= price)
        expected_rewards /= n_simulations
        expected_revenue = expected_rewards * PRICES
        best_indices = np.argmax(expected_revenue, axis=1)
        best_prices = {pid: PRICES[best_indices[pid]] for pid in range(N_PRODUCTS)}
        best_per_round = sum(expected_revenue[pid, best_indices[pid]] for pid in range(N_PRODUCTS))
        return best_prices, best_per_round

    # Compute oracle reward & regret
    oracle_prices, oracle_expected_per_round = estimate_best_fixed_prices()
    oracle_rewards = [oracle_expected_per_round] * len(rewards_log)
    cumulative_oracle = np.cumsum(oracle_rewards)
    cumulative_algo = np.cumsum(rewards_log)
    cumulative_regret = cumulative_oracle - cumulative_algo
    average_regret = cumulative_regret / np.arange(1, len(cumulative_regret) + 1)

    # --- Plot: Cumulative Rewards + Regret ---
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, cumulative_algo, label="Cumulative Primal-Dual Reward")
    plt.plot(rounds, cumulative_oracle, label="Cumulative Oracle Reward", color='orange')
    plt.plot(rounds, cumulative_regret, label="Cumulative Regret", linestyle="--", color="red")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{algo_name} vs Oracle: Reward and Regret")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Plot: Average Regret ---
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, average_regret, color='purple', label="Average Regret R(T)/T")
    plt.xlabel("Round")
    plt.ylabel("Average Regret")
    plt.title(f"Average Regret per Round - {algo_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def main():
    print("📊 Requirement 4: Best-of-Both-Worlds with Multiple Products")
    print("   Algorithm: Primal-Dual")
    print("   Environments: Stochastic & Highly Non-Stationary\n")

    # Stochastic Environment
    stochastic_env = MultiProductStochasticEnvironment(
        n_products=N_PRODUCTS,
        prices=PRICES,
        production_capacity=INVENTORY,
        total_rounds=ROUNDS,
        valuation_distribution="uniform",
        valuation_params={"low": 0.0, "high": 1.0},
        random_seed=42
    )
    run_experiment(stochastic_env, "Primal-Dual (Stochastic)", "Requirement 4 - Stochastic Environment")

    # Highly Non-Stationary Environment
    non_stationary_env = MultiProductHighlyNonStationaryEnvironment(
        n_products=N_PRODUCTS,
        prices=PRICES,
        production_capacity=INVENTORY,
        total_rounds=ROUNDS,
        random_seed=42
    )
    run_experiment(non_stationary_env, "Primal-Dual (Non-Stationary)", "Requirement 4 - Non-Stationary Environment")
    
    


if __name__ == "__main__":
    main()
