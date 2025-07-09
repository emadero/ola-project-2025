
import numpy as np

class ThompsonSamplingMultipleProducts:
    def __init__(self, n_products, prices):
        self.n_products = n_products
        self.prices = prices
        self.n_prices = len(prices)
        self.alpha = np.ones((n_products, self.n_prices))
        self.beta = np.ones((n_products, self.n_prices))

    def select_prices(self):
        selected = {}
        for i in range(self.n_products):
            theta_samples = np.random.beta(self.alpha[i], self.beta[i])
            best_arm = np.argmax(theta_samples)
            selected[i] = self.prices[best_arm]
        return selected

    def update(self, selected_prices, rewards):
        for i in range(self.n_products):
            price = selected_prices[i]
            reward = rewards[i]
            arm = self.prices.index(price)
            self.alpha[i, arm] += reward
            self.beta[i, arm] += 1 - reward
