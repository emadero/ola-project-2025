import numpy as np

class ThompsonSamplingCUSUM:
    def __init__(self, n_products, prices, delta=None, cusum_threshold=None):
        self.n_products = n_products
        self.prices = prices
        self.n_arms = len(prices)
        self.alpha = np.ones((n_products, self.n_arms))
        self.beta = np.ones((n_products, self.n_arms))

        self.delta = delta
        self.cusum_threshold = cusum_threshold
        self.reset_stats()
        self.cumulative_regret = 0.0
        self.last_actions = {}

    def reset_stats(self):
        self.cusum_stats = np.zeros(self.n_products)
        self.empirical_means = np.zeros(self.n_products)
        self.change_detected = np.array([False] * self.n_products)
        self.recent_rewards = [[] for _ in range(self.n_products)]

    def act(self):
        self.last_actions = {}
        for i in range(self.n_products):
            theta = np.random.beta(self.alpha[i], self.beta[i])
            self.last_actions[i] = self.prices[np.argmax(theta)]
        return self.last_actions

    def update(self, selected_prices, rewards):
        for i in range(self.n_products):
            price = selected_prices[i]
            reward = rewards[i]
            arm = self.prices.index(price)

            self.alpha[i, arm] += reward
            self.beta[i, arm] += 1 - reward

            if self.delta is not None and self.cusum_threshold is not None:
                self.recent_rewards[i].append(reward)
                if len(self.recent_rewards[i]) > 25:
                    self.recent_rewards[i].pop(0)
                mean_est = np.mean(self.recent_rewards[i])
                self.cusum_stats[i] += reward - mean_est - self.delta
                if self.cusum_stats[i] > self.cusum_threshold:
                    self.alpha[i] = np.ones(self.n_arms)
                    self.beta[i] = np.ones(self.n_arms)
                    self.cusum_stats[i] = 0
                    self.recent_rewards[i] = []

    def track_regret(self, rewards, valuations):
        for i in range(self.n_products):
            valuation = valuations[i]
            optimal = max([price for price in self.prices if price <= valuation], default=0.0)
            actual_price = self.last_actions[i]
            actual_reward = actual_price if valuation >= actual_price else 0.0
            self.cumulative_regret += optimal - actual_reward

    def get_cumulative_regret(self):
        return self.cumulative_regret
