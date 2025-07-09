# experiments/requirement2.py
"""
Requirement 2: Multiple Products & Stochastic Environment
Assigned to: Maxence Guyot

This experiment implements the Combinatorial UCB1 algorithm for multi-product pricing
in a stochastic environment with joint buyer valuations.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add root directory to sys.path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.multiple_products.combinatorial_ucb import CombinatorialUCB1Algorithm
from environments.multi_stochastic import MultiProductStochasticEnvironment

def main():
    print("🎯 Requirement 2: Combinatorial UCB with Multi-Product Stochastic Environment")
    print("=" * 60)

    # --- Configuration ---
    N_PRODUCTS = 5
    PRICES = [0.2, 0.3, 0.4, 0.5, 0.6]
    INVENTORY = 500
    ROUNDS = 175

    # --- Initialize environment ---
    env = MultiProductStochasticEnvironment(
        n_products=N_PRODUCTS,
        prices=PRICES,
        production_capacity=INVENTORY,
        total_rounds=ROUNDS,
        valuation_distribution="uniform",
        valuation_params={"low": 0.0, "high": 1.0},
        random_seed=42
    )

    # --- Initialize algorithm ---
    algo = CombinatorialUCB1Algorithm(
        n_products=N_PRODUCTS,
        prices=PRICES
    )

    env.reset()

    # --- Logs for plotting ---
    total_revenue = 0
    inventory_used = 0
    rewards_log = []
    cumulative_inventory = []
    rounds = []

    # --- Run simulation ---
    for t in range(ROUNDS):
        prices = algo.select_prices()
        buyer_info, rewards, done = env.step(prices)
        algo.update(prices, rewards)

        round_revenue = sum(rewards.values())
        total_revenue += round_revenue
        inventory_used += sum(buyer_info['purchases'].values())

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

    # --- Visualization ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(rounds, cumulative_inventory, label="Inventory Used")
    axs[0].axhline(y=INVENTORY, color='r', linestyle='--', label="Inventory Limit")
    axs[0].set_ylabel("Units")
    axs[0].set_title("Inventory Usage Over Time")
    axs[0].legend()

    axs[1].plot(rounds, rewards_log, label="Revenue per Round")
    axs[1].set_xlabel("Round")
    axs[1].set_ylabel("Revenue")
    axs[1].set_title("Revenue per Round")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    final_ucbs = algo.get_final_ucbs()

    fig, axs = plt.subplots(1, N_PRODUCTS, figsize=(15, 4))
    for pid in range(N_PRODUCTS):
        bars = axs[pid].bar([str(p) for p in PRICES], final_ucbs[pid])
        
        axs[pid].set_title(f"Product {pid +1} - Final UCBs")
        axs[pid].set_xlabel("Price")
        axs[pid].set_ylabel("UCB")
        axs[pid].grid(True, linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            axs[pid].text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.2f}', ha='center', fontsize=8)


    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
