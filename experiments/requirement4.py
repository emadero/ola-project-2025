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

# Make project modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.multi_stochastic import MultiProductStochasticEnvironment
from environments.multi_non_stationary import MultiProductHighlyNonStationaryEnvironment
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts

# Shared experiment configuration
N_PRODUCTS = 10
PRICES = [0.2, 0.3, 0.4, 0.5, 0.6]
INVENTORY = 500
ROUNDS = 300

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

    for t in range(ROUNDS):
        prices, indices = algo.select_prices()
        buyer_info, rewards, done = env.step(prices)
        algo.update(indices, buyer_info["purchases"], rewards)

        round_revenue = sum(rewards.values())
        total_revenue += round_revenue
        inventory_used += sum(buyer_info["purchases"].values())

        print(f"\nRound {t + 1}")
        print(f"  Selected prices: {prices}")
        print(f"  Buyer valuations: {buyer_info['valuations']}")
        print(f"  Purchases: {buyer_info['purchases']}")
        print(f"  Rewards: {rewards}")
        print(f"  Revenue this round: {round_revenue:.2f}")
        print(f"  Remaining inventory: {env.remaining_inventory}")

        rewards_log.append(round_revenue)
        cumulative_inventory.append(inventory_used)
        rounds.append(t + 1)

        if done:
            print("\n🛑 Environment finished.")
            break

    print(f"\n💰 Total Revenue: {total_revenue:.2f}")

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
