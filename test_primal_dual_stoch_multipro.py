# test_primal_dual_stochastic.py

from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts
from environments.multi_stochastic import MultiProductStochasticEnvironment
import matplotlib.pyplot as plt

print("🧪 Testing Primal-Dual Algorithm (per-product) with Multi-Product Stochastic Environment")
print("=" * 60)

# Initialise l’environnement
env = MultiProductStochasticEnvironment(
    n_products=3,
    prices=[0.2, 0.3, 0.4, 0.5, 0.6],
    production_capacity=50,
    total_rounds=30,
    valuation_distribution='uniform',
    valuation_params={'low': 0.0, 'high': 1.0},
    random_seed=42
)

# Initialise l’algorithme
agent = PrimalDualMultipleProducts(
    price_candidates=[0.2, 0.3, 0.4, 0.5, 0.6],
    n_products=3,
    inventory=50,
    n_rounds=30,
    learning_rate=0.01
)

# Logs
rewards_log = []
inventory_used = 0
cumulative_inventory = []
rounds = []

# Run
state = env.reset()
done = False

while not done:
    prices, indices = agent.select_prices()
    buyer_info, rewards, done = env.step(prices)
    agent.update(indices, buyer_info["purchases"], rewards)

    round_reward = sum(rewards.values())
    rewards_log.append(round_reward)

    inventory_used += sum(buyer_info["purchases"].values())
    cumulative_inventory.append(inventory_used)
    rounds.append(env.current_round)

    print(f"Round {env.current_round}: prices = {prices}, reward = {round_reward:.2f}, inventory used = {inventory_used}")

print("\n🛑 Environment finished.")
print(f"\n💰 Total Revenue: {sum(rewards_log):.2f}")

# 📈 Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

axs[0].plot(rounds, cumulative_inventory, label="Cumulative Inventory Used")
axs[0].axhline(y=50, color='r', linestyle='--', label="Inventory Limit")
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
