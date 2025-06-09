# environments/multi_non_stationary.py
"""
Highly Non-Stationary Environment for Multi-Product Pricing
Assigned to: Maxence (Person 2)

This module implements a non-stationary environment where buyer valuations
are correlated across products and evolve rapidly over time.

The valuations are generated from a latent variable that follows a
sinusoidal trend with added noise. Each product has a valuation that is
a linear transformation of this latent variable plus product-specific noise.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from environments import BaseEnvironment, Buyer


class MultiProductBuyer(Buyer):
    """
    Buyer with valuations for multiple products
    """
    def __init__(self, valuations: Dict[int, float]):
        self.valuations = valuations

    def get_valuation(self, product_id: int) -> float:
        return self.valuations.get(product_id, 0.0)


class MultiProductHighlyNonStationaryEnvironment(BaseEnvironment):
    """
    Highly non-stationary environment for multi-product pricing

    A latent global factor z_t drives all product valuations,
    introducing correlation and fast-changing preferences.
    """

    def __init__(self,
                 n_products: int,
                 prices: List[float],
                 production_capacity: int,
                 total_rounds: int = 1000,
                 random_seed: Optional[int] = None):
        """
        Initialize the non-stationary environment

        Args:
            n_products: Number of products (N)
            prices: List of available prices
            production_capacity: Total inventory budget (B)
            total_rounds: Number of simulation rounds (T)
            random_seed: Seed for reproducibility
        """
        super().__init__(n_products=n_products, prices=prices, production_capacity=production_capacity)

        self.total_rounds = total_rounds
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.remaining_inventory = production_capacity
        self.current_round = 0
        self.z_t = 0.5  # latent factor
        self.t = 0

        # Product-specific sensitivity to z_t
        self.a = np.linspace(0.8, 1.2, n_products)

        print(f"⚡️ Highly Non-Stationary Multi-Product Environment Initialized")
        print(f"   📦 Products: {n_products}")
        print(f"   🎲 Prices: {min(prices)} to {max(prices)}")
        print(f"   🧠 Latent dynamics: sinusoidal + noise")
        print(f"   🏭 Inventory: {production_capacity}")
        print(f"   🔄 Rounds: {total_rounds}")

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state

        Returns:
            Initial state dict
        """
        self.current_round = 0
        self.remaining_inventory = self.production_capacity
        self.z_t = 0.5
        self.t = 0
        return {"round": 0, "inventory": self.remaining_inventory}

    def _generate_buyer(self) -> MultiProductBuyer:
        """
        Generate a buyer with correlated and evolving valuations

        Returns:
            Buyer instance with per-product valuations
        """
        # Evolve the latent factor z_t
        self.z_t = 0.5 + 0.4 * np.sin(2 * np.pi * self.t / 25) + self.rng.normal(0, 0.05)

        # Product valuations = a_i * z_t + noise
        valuations = {}
        for i in range(self.n_products):
            noise = self.rng.normal(0, 0.05)
            val = self.a[i] * self.z_t + noise
            valuations[i] = float(np.clip(val, 0.0, 1.0))  # keep in [0, 1]

        self.t += 1
        return MultiProductBuyer(valuations)

    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict[str, Any], Dict[int, float], bool]:
        """
        Execute one round of the pricing game

        Args:
            selected_prices: Dict mapping product_id -> offered price

        Returns:
            Tuple (buyer_info, rewards, done)
        """
        if self.remaining_inventory <= 0 or self.current_round >= self.total_rounds:
            return {}, {}, True

        self.current_buyer = self._generate_buyer()
        buyer = self.current_buyer

        purchases = buyer.make_purchases(selected_prices)

        rewards = {}
        for pid, bought in purchases.items():
            if bought and self.remaining_inventory > 0:
                rewards[pid] = selected_prices[pid]
                self.remaining_inventory -= 1
            else:
                rewards[pid] = 0.0

        buyer_info = {
            "round": self.current_round,
            "valuations": buyer.valuations,
            "purchases": purchases
        }

        self.current_round += 1
        done = self.remaining_inventory <= 0 or self.current_round >= self.total_rounds

        return buyer_info, rewards, done

    def get_buyer_valuations(self) -> Dict[int, float]:
        """
        Return current buyer valuations

        Returns:
            Dict product_id -> valuation
        """
        if not hasattr(self, "current_buyer") or self.current_buyer is None:
            return {}
        return self.current_buyer.valuations

def demo_environment():
    """
    Demo for the MultiProductHighlyNonStationaryEnvironment
    """
    print("🎬 Demo: Highly Non-Stationary Multi-Product Environment")
    print("=" * 50)

    env = MultiProductHighlyNonStationaryEnvironment(
        n_products=2,
        prices=[0.3, 0.4, 0.5],
        production_capacity=10,
        total_rounds=10,
        random_seed=123
    )

    env.reset()

    for t in range(10):
        selected_prices = {0: 0.6, 1: 0.7, 2: 0.8}
        buyer_info, rewards, done = env.step(selected_prices)

        print(f"Round {t + 1}:")
        print(f"  Valuations: {buyer_info['valuations']}")
        print(f"  Purchases:  {buyer_info['purchases']}")
        print(f"  Rewards:    {rewards}")
        print()

        if done:
            print("🔚 Environment finished.")
            break

if __name__ == "__main__":
    demo_environment()
