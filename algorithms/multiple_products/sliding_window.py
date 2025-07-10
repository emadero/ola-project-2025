# algorithms/multiple_products/sliding_window.py

import numpy as np
from collections import deque
from typing import List, Dict


class SlidingWindowCUCB:
    """
    Sliding Window Combinatorial UCB for Multi-Product Pricing
    """

    def __init__(self, n_products: int, prices: List[float], window_size: int = 100):
        self.n_products = n_products
        self.prices = prices
        self.n_prices = len(prices)
        self.window_size = window_size
        self.time = 0

        # For each (product, price), store recent rewards (as deque)
        self.windows = [[deque(maxlen=window_size) for _ in range(self.n_prices)]
                        for _ in range(n_products)]

    def select_prices(self) -> Dict[int, float]:
        self.time += 1
        selected_prices = {}

        for i in range(self.n_products):
            ucb_values = []
            for j, _ in enumerate(self.prices):
                history = self.windows[i][j]
                n = len(history)
                if n == 0:
                    ucb = float('inf')  # force exploration
                else:
                    mean = np.mean(history)
                    bonus = np.sqrt((2 * np.log(self.time)) / n)
                    ucb = mean + bonus
                ucb_values.append(ucb)

            best_j = int(np.argmax(ucb_values))
            selected_prices[i] = self.prices[best_j]

        return selected_prices

    def update(self, chosen_prices: Dict[int, float], rewards: Dict[int, float]):
        for i, price in chosen_prices.items():
            j = self.prices.index(price)
            self.windows[i][j].append(rewards.get(i, 0.0))
