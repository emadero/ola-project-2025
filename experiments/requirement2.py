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
import numpy as np

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
    INVENTORY = 5000
    ROUNDS = 1750

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


    # --- Oracle computation: estimate best fixed prices over multiple simulations ---
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

    # Compute best fixed prices and regret
    oracle_prices, oracle_expected_revenue_per_round = estimate_best_fixed_prices()
    oracle_revenues = [oracle_expected_revenue_per_round] * len(rewards_log)
    regret = np.cumsum(np.array(oracle_revenues) - np.array(rewards_log))

    print(f"💰 UCB1 revenue per round (estimated): {total_revenue/len(rewards_log) :.3f}")
    print(f"\n🔍 Oracle revenue per round (estimated): {oracle_expected_revenue_per_round:.3f}")
    print(f"📉 Total regret after {len(rewards_log)} rounds: {regret[-1]:.3f}")
    
    simulate_ucb_vs_oracle(rewards_log, env, PRICES, ROUNDS)

def compute_oracle_rewards(env, prices, rounds):
    """
    Simule les meilleures actions fixes (oracle) pour chaque produit.
    Retourne la courbe de reward cumulatif.
    """
    best_combo = {}

    # Estimation empirique : simulate for each product and price
    estimated_rewards = {}
    for pid in range(env.n_products):
        best_price = None
        best_expected_reward = 0
        for p in prices:
            total = 0
            for _ in range(1000):  # simulate 1000 buyers
                buyer = env._generate_buyer()
                if buyer.valuations[pid] >= p:
                    total += p
            expected = total / 1000
            if expected > best_expected_reward:
                best_expected_reward = expected
                best_price = p
        best_combo[pid] = best_price

    # Simulate rounds with best prices
    env.reset()
    oracle_rewards = []
    total = 0
    for _ in range(rounds):
        buyer = env._generate_buyer()
        rewards = 0
        allowed = env.remaining_inventory
        for pid in range(env.n_products):
            if buyer.valuations[pid] >= best_combo[pid] and allowed > 0:
                rewards += best_combo[pid]
                allowed -= 1
        total += rewards
        oracle_rewards.append(total)
    return oracle_rewards

def simulate_ucb_vs_oracle(rewards_log, env, prices, rounds):
    """
    Generate cumulative reward curves for UCB1 and oracle, and compute regret.
    """
    cumulative_ucb_reward = np.cumsum(rewards_log)
    oracle_rewards = compute_oracle_rewards(env, prices, len(rewards_log))
    cumulative_oracle_reward = np.array(oracle_rewards)
    regret = cumulative_oracle_reward - cumulative_ucb_reward

    x_axis = np.arange(1, len(regret) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, cumulative_ucb_reward, label="Cumulative UCB1 Reward")
    plt.plot(x_axis, cumulative_oracle_reward, label="Cumulative Oracle Reward")
    plt.plot(x_axis, regret, label="Cumulative Regret", linestyle="--", color="red")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title("UCB1 vs Oracle: Reward and Regret")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Plot average regret R(T)/T ---
    avg_regret = regret / np.arange(1, len(regret) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(regret) + 1), avg_regret, label="Average Regret R(T)/T", color="purple")
    plt.xlabel("Round")
    plt.ylabel("Average Regret")
    plt.title("Average Regret per Round")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return cumulative_oracle_reward, regret


if __name__ == "__main__":
    main()
