# algorithms/multiple_products/primal_dual.py

import numpy as np

class PerProductPrimalDual:
    """
    Primal-Dual regret minimizer for a single product.
    Maintains a distribution over price options via exponential weights (Hedge).
    """

    def __init__(self, price_candidates, learning_rate=0.1):
        self.price_candidates = price_candidates
        self.n_prices = len(price_candidates)
        self.learning_rate = learning_rate

        # Initialize weights for Hedge
        self.weights = np.ones(self.n_prices)
        self.probs = self.weights / np.sum(self.weights)

    def select_price(self):
        """
        Sample a price according to the current distribution
        """
        idx = np.random.choice(self.n_prices, p=self.probs)
        return self.price_candidates[idx], idx

    def update(self, chosen_idx, reward, dual_penalty):
        """
        Perform Hedge-style update using the received reward minus dual penalty.
        """
        # Compute gain (primal objective minus dual cost)
        gains = np.zeros(self.n_prices)
        gains[chosen_idx] = reward - dual_penalty

        # Exponentiated gradient (Hedge)
        self.weights *= np.exp(self.learning_rate * gains)
        self.probs = self.weights / np.sum(self.weights)
        
        # Sanity check to avoid NaNs
        if np.any(np.isnan(self.probs)) or np.sum(self.probs) == 0:
            # Fallback to uniform
            self.probs = np.ones_like(self.weights) / len(self.weights)


class PrimalDualMultipleProducts:
    """
    Fully decomposed primal-dual algorithm.
    Each product uses its own regret minimizer to select prices.
    A global dual variable handles the inventory constraint.
    """

    def __init__(self, price_candidates, n_products, inventory, n_rounds, learning_rate=0.1):
        self.n_products = n_products
        self.price_candidates = price_candidates
        self.n_rounds = n_rounds
        self.inventory = inventory
        self.learning_rate = learning_rate

        # One learner per product
        self.learners = [
            PerProductPrimalDual(price_candidates, learning_rate)
            for _ in range(n_products)
        ]

        self.lambda_dual = 0.0  # shared dual variable (inventory multiplier)

        self.history = {
            "selected_prices": [],
            "purchases": [],
            "rewards": [],
            "revenues": [],
        }

    def select_prices(self):
        """
        Select a price for each product using individual regret minimizers.
        """
        prices = {}
        indices = {}
        for pid, learner in enumerate(self.learners):
            price, idx = learner.select_price()
            prices[pid] = price
            indices[pid] = idx
        return prices, indices

    def update(self, price_indices, purchases, rewards):
        """
        Update dual variable and per-product learners.
        """
        consumption = sum(purchases.values())
        gradient = consumption - (self.inventory / self.n_rounds)
        self.lambda_dual += self.learning_rate * gradient
        self.lambda_dual = max(0.0, self.lambda_dual)

        for pid in range(self.n_products):
            chosen_idx = price_indices[pid]
            reward = rewards[pid]
            self.learners[pid].update(chosen_idx, reward, self.lambda_dual)

    def run(self, environment):
        """
        Run the algorithm in the environment.
        """
        state = environment.reset()
        done = False

        while not done:
            prices, price_indices = self.select_prices()
            buyer_info, rewards, done = environment.step(prices)
            purchases = buyer_info["purchases"]

            self.update(price_indices, purchases, rewards)

            self.history["selected_prices"].append(prices)
            self.history["purchases"].append(purchases)
            self.history["rewards"].append(rewards)
            self.history["revenues"].append(sum(rewards.values()))

        return self.history
