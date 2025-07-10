import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import deque

# ---------------------------
# ENVIRONMENT
# ---------------------------

class MultiProductBuyer:
    def __init__(self, valuations):
        self.valuations = valuations

    def make_purchases(self, prices):
        return {pid: prices[pid] <= self.valuations[pid] for pid in prices}


class MultiProductPiecewiseStationaryEnvironment:
    def __init__(self, n_products, prices, production_capacity,
                 total_rounds, n_intervals, random_seed=42):
        self.n_products = n_products
        self.prices = prices
        self.production_capacity = production_capacity
        self.total_rounds = total_rounds
        self.n_intervals = n_intervals
        self.rng = np.random.RandomState(random_seed)
        self.interval_length = total_rounds // n_intervals
        self.valuation_distributions = self._generate_distributions()
        self.reset()

    def _generate_distributions(self):
        dists = []
        for _ in range(self.n_intervals):
            dists.append({
                pid: (self.rng.uniform(0.0, 0.5),
                      self.rng.uniform(0.5, 1.0))
                for pid in range(self.n_products)
            })
        return dists

    def reset(self):
        self.current_round = 0
        self.remaining_inventory = self.production_capacity

    def _generate_buyer(self):
        idx = min(self.current_round // self.interval_length,
                  self.n_intervals - 1)
        return MultiProductBuyer({
            pid: self.rng.uniform(*self.valuation_distributions[idx][pid])
            for pid in range(self.n_products)
        })

    def step(self, prices):
        if self.remaining_inventory <= 0 or self.current_round >= self.total_rounds:
            return None, {}, True

        buyer = self._generate_buyer()
        purchases = buyer.make_purchases(prices)
        rewards = {}
        for pid, bought in purchases.items():
            if bought and self.remaining_inventory > 0:
                rewards[pid] = prices[pid]
                self.remaining_inventory -= 1
            else:
                rewards[pid] = 0.0

        self.current_round += 1
        done = (self.remaining_inventory <= 0 or
                self.current_round >= self.total_rounds)
        return {"purchases": purchases}, rewards, done

    def get_buyer_valuations(self):
        return self._generate_buyer().valuations
# ---------------------------
# ALGORITHMS
# ---------------------------

class SlidingWindowCUCB:
    def __init__(self, n_products, prices, window_size,
                 delta=None, cusum_threshold=None):
        self.n_products = n_products
        self.prices = prices
        self.window_size = window_size
        self.delta = delta
        self.threshold = cusum_threshold
        self.reset()

    def reset(self):
        self.t = 0
        self.windows = [
            [deque(maxlen=self.window_size) for _ in self.prices]
            for _ in range(self.n_products)
        ]
        self.stats = [0.0] * self.n_products
        self.change = [False] * self.n_products
        self._cumulative_regret = 0.0

    def select_prices(self):
        self.t += 1
        selected = {}
        for i in range(self.n_products):
            ucbs = []
            for j in range(len(self.prices)):
                hist = self.windows[i][j]
                if not hist or self.change[i]:
                    ucbs.append(float('inf'))
                else:
                    ucbs.append(
                        np.mean(hist) +
                        np.sqrt((2 * np.log(self.t)) / len(hist))
                    )
            selected[i] = self.prices[np.argmax(ucbs)]
        return selected

    def update(self, selected_prices, rewards):
        for i, price in selected_prices.items():
            j = self.prices.index(price)
            r = rewards[i]
            self.windows[i][j].append(r)
            if self.delta is not None and self.threshold is not None:
                self.stats[i] += r - self.delta
                if self.stats[i] > self.threshold:
                    self.windows[i] = [
                        deque(maxlen=self.window_size) for _ in self.prices
                    ]
                    self.stats[i] = 0.0
                    self.change[i] = True
                else:
                    self.change[i] = False

    def track_regret(self, rewards, valuations):
        for p in rewards:
            optimal = max([price for price in self.prices if price <= valuations[p]], default=0.0)
            self._cumulative_regret += optimal - rewards[p]

    def get_cumulative_regret(self):
        return self._cumulative_regret


class ThompsonSamplingSlidingWindow:
    def __init__(self, n_products, prices, window_size=50):
        self.n_products = n_products
        self.prices = prices
        self.window_size = window_size
        self.rewards = [[[] for _ in prices] for _ in range(n_products)]
        self.last_actions = {}
        self._cumulative_regret = 0.0

    def select_prices(self):
        selected = {}
        for i in range(self.n_products):
            samples = []
            for j in range(len(self.prices)):
                h = self.rewards[i][j]
                alpha = 1 + sum(h)
                beta = 1 + len(h) - sum(h)
                samples.append(np.random.beta(alpha, beta))
            best = np.argmax(samples)
            selected[i] = self.prices[best]
            self.last_actions[i] = self.prices[best]
        return selected

    def update(self, selected_prices, rewards):
        for i, price in selected_prices.items():
            j = self.prices.index(price)
            self.rewards[i][j].append(rewards[i])
            if len(self.rewards[i][j]) > self.window_size:
                self.rewards[i][j].pop(0)

    def track_regret(self, rewards, valuations):
        for i in range(self.n_products):
            optimal = max([p for p in self.prices if p <= valuations[i]], default=0.0)
            actual = self.last_actions[i] if valuations[i] >= self.last_actions[i] else 0.0
            self._cumulative_regret += optimal - actual

    def get_cumulative_regret(self):
        return self._cumulative_regret


class ThompsonSamplingCUSUM(ThompsonSamplingSlidingWindow):
    def __init__(self, n_products, prices, delta, threshold):
        super().__init__(n_products, prices)
        self.delta = delta
        self.threshold = threshold
        self.stats = [0.0] * n_products
        self.recent = [[] for _ in range(n_products)]

    def update(self, selected_prices, rewards):
        super().update(selected_prices, rewards)
        for i, price in selected_prices.items():
            j = self.prices.index(price)
            self.recent[i].append(rewards[i])
            if len(self.recent[i]) > 25:
                self.recent[i].pop(0)
            mean_est = np.mean(self.recent[i]) if self.recent[i] else 0
            self.stats[i] += rewards[i] - mean_est - self.delta
            if self.stats[i] > self.threshold:
                self.rewards[i] = [[] for _ in self.prices]
                self.stats[i] = 0.0
                self.recent[i] = []


class PerProductPrimalDual:
    def __init__(self, price_candidates, learning_rate=0.1):
        self.price_candidates = price_candidates
        self.learning_rate = learning_rate
        self.weights = np.ones(len(price_candidates))

    def select_price(self):
        probs = self.weights / np.sum(self.weights)
        idx = np.random.choice(len(self.price_candidates), p=probs)
        return self.price_candidates[idx], idx

    def update(self, idx, reward, dual_penalty):
        gain = reward - dual_penalty
        self.weights[idx] *= np.exp(self.learning_rate * gain)


class PrimalDualMultipleProducts:
    def __init__(self, price_candidates, n_products,
                 inventory, n_rounds, learning_rate=0.01):
        self.learners = [
            PerProductPrimalDual(price_candidates, learning_rate)
            for _ in range(n_products)
        ]
        self.lambda_dual = 0.0
        self.n_products = n_products
        self.inventory = inventory
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate

    def select_prices(self):
        prices = {}
        indices = {}
        for pid, learner in enumerate(self.learners):
            price, idx = learner.select_price()
            prices[pid] = price
            indices[pid] = idx
        return prices, indices

    def update(self, indices, purchases, rewards):
        consumption = sum(purchases.values())
        grad = consumption - (self.inventory / self.n_rounds)
        self.lambda_dual = max(0.0, self.lambda_dual +
                               self.learning_rate * grad)
        for pid in range(self.n_products):
            self.learners[pid].update(
                indices[pid], rewards[pid], self.lambda_dual
            )

    def run(self, env):
        revs, regs = [], []
        env.reset()
        done = False
        while not done:
            prices, indices = self.select_prices()
            buyer_info, reward, done = env.step(prices)
            self.update(indices, buyer_info["purchases"], reward)
            r = sum(reward.values())
            revs.append(r)
            regs.append(8.0 - r)
        return np.cumsum(revs), np.cumsum(regs)
# ---------------------------
# RUN & PLOT
# ---------------------------

n_products = 8
prices = [round(0.1 * i, 1) for i in range(1, 11)]
env_cfg = dict(
    n_products=n_products,
    prices=prices,
    production_capacity=500 * n_products,
    total_rounds=1000,
    n_intervals=8,
    random_seed=42
)

agents = [
    ("SW", SlidingWindowCUCB(n_products, prices, window_size=20)),
    ("SW-CUSUM", SlidingWindowCUCB(n_products, prices, window_size=20,
                                   delta=0.03, cusum_threshold=1.02)),
    ("TS-SW", ThompsonSamplingSlidingWindow(n_products, prices,
                                            window_size=50)),
    ("TS-CUSUM", ThompsonSamplingCUSUM(n_products, prices,
                                       delta=0.005, threshold=1.07)),
]

results = {}
for name, agent in agents:
    if hasattr(agent, 'reset'):
        agent.reset()
    env = MultiProductPiecewiseStationaryEnvironment(**env_cfg)
    rev, reg = [], []
    done = False
    while not done:
        selected = agent.select_prices()
        _, reward, done = env.step(selected)
        valuations = env.get_buyer_valuations()
        if hasattr(agent, 'track_regret'):
            agent.track_regret(reward, valuations)
        agent.update(selected, reward)
        rev.append(sum(reward.values()))
        if hasattr(agent, 'get_cumulative_regret'):
            reg.append(agent.get_cumulative_regret())
    results[name] = (np.cumsum(rev), np.array(reg))

# Add PrimalDual
primal = PrimalDualMultipleProducts(
    prices, n_products,
    env_cfg["production_capacity"],
    env_cfg["total_rounds"]
)
pd_env = MultiProductPiecewiseStationaryEnvironment(**env_cfg)
results["PRIMAL"] = primal.run(pd_env)

# Save folder
os.makedirs("results/final_comparison", exist_ok=True)

# Plot Revenue
plt.figure(figsize=(10, 5))
for name, (rev, _) in results.items():
    plt.plot(rev, label=name)
plt.title("Cumulative Revenue")
plt.xlabel("Round")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.savefig("results/final_comparison/revenue.png")
plt.show()

# Plot Regret
plt.figure(figsize=(10, 5))
for name, (_, reg) in results.items():
    if reg is not None and len(reg) > 0:
        plt.plot(reg, label=name)
plt.title("Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Regret")
plt.legend()
plt.grid(True)
plt.savefig("results/final_comparison/regret.png")
plt.show()

# Heatmaps
final_regret = {name: reg[-1] if len(reg) > 0 else None for name, (_, reg) in results.items()}
df_regret = pd.DataFrame.from_dict(final_regret, orient="index", columns=["Final Regret"]).sort_values("Final Regret")
plt.figure(figsize=(7, 4))
sns.heatmap(df_regret.T, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True)
plt.title("Final Cumulative Regret by Algorithm")
plt.tight_layout()
plt.savefig("results/final_comparison/heatmap_regret.png")
plt.show()

final_revenue = {name: rev[-1] for name, (rev, _) in results.items()}
df_revenue = pd.DataFrame.from_dict(final_revenue, orient="index", columns=["Final Revenue"]).sort_values("Final Revenue", ascending=False)
plt.figure(figsize=(7, 4))
sns.heatmap(df_revenue.T, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
plt.title("Final Cumulative Revenue by Algorithm")
plt.tight_layout()
plt.savefig("results/final_comparison/heatmap_revenue.png")
plt.show()

# Bar charts
plt.figure(figsize=(8, 5))
sns.barplot(x=df_regret.index, y="Final Regret", data=df_regret.reset_index())
plt.title("Final Cumulative Regret by Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Final Regret")
plt.tight_layout()
plt.savefig("results/final_comparison/bar_regret.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=df_revenue.index, y="Final Revenue", data=df_revenue.reset_index())
plt.title("Final Cumulative Revenue by Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Final Revenue")
plt.tight_layout()
plt.savefig("results/final_comparison/bar_revenue.png")
plt.show()
