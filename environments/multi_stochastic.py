# environments/multi_stochastic.py
"""
Stochastic Environment for Multi-Product Pricing
Assigned to: Maxence (Person 2)

This module extends the single-product stochastic environment to multiple product types.
Each buyer arrives with a valuation for each product. The buyer purchases all products
where the offered price is below their valuation. The environment enforces a global 
inventory constraint across all products.
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


class MultiProductStochasticEnvironment(BaseEnvironment):
    """
    Stochastic environment for multiple product pricing

    In each round, a buyer arrives with a vector of valuations, one for each product.
    The company sets a price for each product, and the buyer purchases all products
    priced below their respective valuations. There is a shared production capacity
    across all products (B units total).
    """

    def __init__(self,
                 n_products: int,
                 prices: List[float],
                 production_capacity: int,
                 total_rounds: int = 1000,
                 valuation_distribution: str = "uniform",
                 valuation_params: Dict[str, float] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the multi-product stochastic environment

        Args:
            n_products: Number of different products (N)
            prices: List of possible prices (shared for all products)
            production_capacity: Total number of products that can be sold (B)
            total_rounds: Number of simulation rounds (T)
            valuation_distribution: Distribution type ("uniform" by default)
            valuation_params: Parameters for the distribution
            random_seed: Random seed for reproducibility
        """
        super().__init__(n_products=n_products, prices=prices, production_capacity=production_capacity)

        self.total_rounds = total_rounds
        self.valuation_distribution = valuation_distribution
        self.random_seed = random_seed
        self.valuation_params = valuation_params or self._default_params()
        self.rng = np.random.RandomState(random_seed)

        self.remaining_inventory = production_capacity

        print(f"🏭 Multi-Product Stochastic Environment initialized:")
        print(f"   📦 Products: {n_products}")
        print(f"   💰 Price range: {min(prices):.2f} - {max(prices):.2f}")
        print(f"   🎲 Distribution: {valuation_distribution}")
        print(f"   ⚙️ Params: {self.valuation_params}")
        print(f"   🏭 Inventory capacity: {production_capacity}")
        print(f"   🔄 Rounds: {total_rounds}")

    def _default_params(self):
        """Return default distribution parameters"""
        return {"low": 0.0, "high": 1.0}

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state

        Returns:
            Initial state dictionary
        """
        self.current_round = 0
        self.remaining_inventory = self.production_capacity
        return {
            "round": self.current_round,
            "inventory": self.remaining_inventory
        }

    def _generate_buyer(self) -> MultiProductBuyer:
        """
        Generate a buyer with valuations for each product

        Returns:
            Buyer instance with per-product valuations
        """
        valuations = {}
        for pid in range(self.n_products):
            if self.valuation_distribution == "uniform":
                val = self.rng.uniform(
                    self.valuation_params["low"],
                    self.valuation_params["high"]
                )
            else:
                raise NotImplementedError("Only uniform distribution is currently supported")
            valuations[pid] = val
        return MultiProductBuyer(valuations)

    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict[str, Any], Dict[int, float], bool]:
        """
        Execute one round of the pricing game

        Args:
            selected_prices: Dict mapping product_id -> price

        Returns:
            Tuple (buyer_info, rewards, done)
        """
        if self.remaining_inventory <= 0 or self.current_round >= self.total_rounds:
            return {}, {}, True  # Simulation is over

        self.current_buyer = self._generate_buyer()
        buyer = self.current_buyer

        purchases = buyer.make_purchases(selected_prices)

        rewards = {}
        units_sold = 0

        for pid, bought in purchases.items():
            if bought and self.remaining_inventory > 0:
                rewards[pid] = selected_prices[pid]
                self.remaining_inventory -= 1
                units_sold += 1
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
        Get current buyer's valuations

        Returns:
            Dict mapping product_id -> buyer's valuation
        """
        if not hasattr(self, "current_buyer") or self.current_buyer is None:
            return {}
        return self.current_buyer.valuations

