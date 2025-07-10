import numpy as np
from collections import deque

class SlidingWindowCUCB:
    def __init__(self, n_products, prices, window_size=10, delta=0.05, cusum_threshold=0.5):
        self.n_products = n_products
        self.prices = prices
        self.window_size = window_size
        self.delta = delta
        self.cusum_threshold = cusum_threshold
        self.n_prices = len(prices)
        self.reset()

    def reset(self):
        self.t = 0
        self.windows = [[deque(maxlen=self.window_size) for _ in range(self.n_prices)]
                        for _ in range(self.n_products)]
        self.cusum_stats = [0.0 for _ in range(self.n_products)]
        self.change_detected = [False for _ in range(self.n_products)]
        self._cumulative_regret = 0.0
        self.reset_counts = [0 for _ in range(self.n_products)]

    def select_prices(self):
        self.t += 1
        selected_prices = {}

        for i in range(self.n_products):
            ucb_values = []
            for j, _ in enumerate(self.prices):
                history = self.windows[i][j]
                n = len(history)
                if n == 0 or self.change_detected[i]:
                    ucb = float('inf')
                else:
                    mean = np.mean(history)
                    bonus = np.sqrt((2 * np.log(self.t)) / n)
                    ucb = mean + bonus
                ucb_values.append(ucb)

            best_j = int(np.argmax(ucb_values))
            selected_prices[i] = self.prices[best_j]

        return selected_prices

    def update(self, chosen_prices, rewards):
        for i, price in chosen_prices.items():
            j = self.prices.index(price)
            reward = rewards.get(i, 0.0)
            self.windows[i][j].append(reward)

            if self.delta is not None and self.cusum_threshold is not None:
                if len(self.windows[i][j]) >= 2:
                    mean_reward = np.mean(self.windows[i][j])
                    deviation = reward - mean_reward - self.delta
                    self.cusum_stats[i] = max(0.0, self.cusum_stats[i] + deviation)

                    if self.cusum_stats[i] > self.cusum_threshold:
                        self.change_detected[i] = True
                        self.cusum_stats[i] = 0.0
                        self.windows[i][j].clear()
                        self.reset_counts[i] += 1
                    else:
                        self.change_detected[i] = False
            else:
                self.change_detected[i] = False

    def track_regret(self, rewards, valuations):
        for p in rewards:
            valuation = valuations[p]
            optimal = max([price for price in self.prices if price <= valuation], default=0.0)
            actual = rewards[p]
            self._cumulative_regret += optimal - actual

    def get_cumulative_regret(self):
        return self._cumulative_regret

    def get_reset_counts(self):
        return self.reset_counts
