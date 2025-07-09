import numpy as np
import matplotlib.pyplot as plt
import os
from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment
from thompson_sampling_cusum import ThompsonSamplingCUSUM

# --- Thompson Sampling with Sliding Window ---
class ThompsonSamplingSlidingWindow:
    def __init__(self, n_products, prices, window_size=50):
        self.n_products = n_products
        self.prices = prices
        self.n_arms = len(prices)
        self.window_size = window_size
        self.rewards_window = [[[] for _ in range(self.n_arms)] for _ in range(n_products)]
        self.last_actions = {}

    def act(self):
        self.last_actions = {}
        for i in range(self.n_products):
            samples = []
            for j in range(self.n_arms):
                rewards = self.rewards_window[i][j]
                alpha = 1 + sum(rewards)
                beta = 1 + len(rewards) - sum(rewards)
                samples.append(np.random.beta(alpha, beta))
            best = np.argmax(samples)
            self.last_actions[i] = self.prices[best]
        return self.last_actions

    def update(self, selected_prices, rewards):
        for i in range(self.n_products):
            price = selected_prices[i]
            reward = rewards[i]
            arm = self.prices.index(price)
            window = self.rewards_window[i][arm]
            window.append(reward)
            if len(window) > self.window_size:
                window.pop(0)

    def track_regret(self, rewards, valuations):
        self.cumulative_regret = getattr(self, "cumulative_regret", 0.0)
        for i in range(self.n_products):
            valuation = valuations[i]
            optimal = max([p for p in self.prices if p <= valuation], default=0.0)
            actual_price = self.last_actions[i]
            actual_reward = actual_price if valuation >= actual_price else 0.0
            self.cumulative_regret += optimal - actual_reward

    def get_cumulative_regret(self):
        return getattr(self, "cumulative_regret", 0.0)

# --- Setup ---
n_products = 8
prices = np.round(np.linspace(0.1, 1.0, 10), 2).tolist()
total_rounds = 1000
production_capacity = 700 * n_products

# --- Parameters ---
delta = 0.04
cusum_threshold = 1.5
window_size = 40

# --- Environment Config ---
config = dict(
    n_products=n_products,
    prices=prices,
    production_capacity=production_capacity,
    total_rounds=total_rounds,
    n_intervals=8,
    random_seed=42
)

# --- TS + CUSUM ---
env1 = MultiProductPiecewiseStationaryEnvironment(**config)
agent1 = ThompsonSamplingCUSUM(
    n_products=n_products,
    prices=prices,
    delta=delta,
    cusum_threshold=cusum_threshold
)
rewards1 = []
done = False
while not done:
    prices_selected = agent1.act()
    _, reward, done = env1.step(prices_selected)
    valuations = env1.get_buyer_valuations()
    agent1.update(prices_selected, reward)
    agent1.track_regret(reward, valuations)
    rewards1.append(sum(reward.values()))

# --- TS + Sliding Window ---
env2 = MultiProductPiecewiseStationaryEnvironment(**config)
agent2 = ThompsonSamplingSlidingWindow(
    n_products=n_products,
    prices=prices,
    window_size=window_size
)
rewards2 = []
done = False
while not done:
    prices_selected = agent2.act()
    _, reward, done = env2.step(prices_selected)
    valuations = env2.get_buyer_valuations()
    agent2.update(prices_selected, reward)
    agent2.track_regret(reward, valuations)
    rewards2.append(sum(reward.values()))

# --- Plot ---
os.makedirs("results/figures/comparisons", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(rewards1), label="TS + CUSUM")
plt.plot(np.cumsum(rewards2), label="TS + Sliding Window", linestyle="--")
plt.title("TS + CUSUM vs TS + Sliding Window")
plt.xlabel("Round")
plt.ylabel("Cumulative Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/comparisons/tssw_vs_tscusum_only.png")
plt.show()
