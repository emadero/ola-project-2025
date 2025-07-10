# algorithms/multiple_products/combinatorial_ucb.py

from typing import List, Dict
import numpy as np

class CombinatorialUCB1Algorithm:
    """
    Combinatorial UCB1 algorithm for multi-product pricing.
    Selects one price per product using UCB strategy.
    """

    def __init__(self, n_products: int, prices: List[float]):
        """
        Initialize the algorithm with all products and their price options.

        Args:
            n_products: Number of products
            prices: List of possible prices (same for all products)
        """
        self.n_products = n_products
        self.prices = prices
        self.n_prices = len(prices)

        # Count of how many times each (product, price) has been selected
        self.counts = np.zeros((n_products, self.n_prices))
        # Cumulative revenue observed for each (product, price)
        self.rewards = np.zeros((n_products, self.n_prices))

        self.total_rounds = 0

    def select_prices(self) -> Dict[int, float]:
        """
        Selects one price per product using UCB1.

        Returns:
            Dictionary {product_id: selected_price}
        """
        selected_prices = {}
        self.total_rounds += 1

        for i in range(self.n_products):
            ucb_values = []

            for j, p in enumerate(self.prices):
                if self.counts[i, j] == 0:
                    # If never tried, force exploration
                    ucb = float('inf')
                else:
                    # Compute average reward and exploration bonus
                    avg_reward = self.rewards[i, j] / self.counts[i, j]
                    bonus = np.sqrt((2 * np.log(self.total_rounds)) / self.counts[i, j])
                    ucb = avg_reward + bonus
                ucb_values.append(ucb)

            # Select price with max UCB for product i
            best_price_index = int(np.argmax(ucb_values))
            selected_prices[i] = self.prices[best_price_index]

        return selected_prices

    def update(self, chosen_prices: Dict[int, float], rewards: Dict[int, float]):
        """
        Update the statistics after observing the rewards.

        Args:
            chosen_prices: Dict {product_id: price chosen}
            rewards: Dict {product_id: observed reward}
        """
        for i, price in chosen_prices.items():
            j = self.prices.index(price)
            self.counts[i, j] += 1
            self.rewards[i, j] += rewards.get(i, 0.0)
            

    def get_final_ucbs(self):
        """
        Return the final UCB values for each product and price index.
        """
        ucbs = {}

        for pid in range(self.n_products):
            product_ucbs = []
            for i in range(self.n_prices):
                pulls = self.counts[pid, i]
                if pulls == 0:
                    mean = 0.0
                    bonus = float("inf")
                else:
                    mean = self.rewards[pid, i] / pulls
                    bonus = np.sqrt((2 * np.log(self.total_rounds)) / pulls)
                product_ucbs.append(mean + bonus)
            ucbs[pid] = product_ucbs

        return ucbs