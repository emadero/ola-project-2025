# environments/slightly_ns.py

from environments import BaseEnvironment, Buyer
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MultiProductBuyer(Buyer):
    def __init__(self, valuations: Dict[int, float]):
        self.valuations = valuations

    def get_valuation(self, product_id: int) -> float:
        return self.valuations.get(product_id, 0.0)


class MultiProductPiecewiseStationaryEnvironment(BaseEnvironment):
    """
    Slightly Non-Stationary Environment for Multi-Product Pricing
    Changes valuation distributions every fixed interval.
    """

    def __init__(self,
                 n_products: int,
                 prices: List[float],
                 production_capacity: int,
                 total_rounds: int = 1000,
                 n_intervals: int = 5,
                 random_seed: Optional[int] = None):
        super().__init__(n_products=n_products, prices=prices, production_capacity=production_capacity)

        self.total_rounds = total_rounds
        self.n_intervals = n_intervals
        self.interval_length = total_rounds // n_intervals
        self.rng = np.random.RandomState(random_seed)
        self.remaining_inventory = production_capacity
        self.current_round = 0
        self.valuation_distributions = self._generate_interval_distributions()

        print("🧩 Piecewise Stationary Environment Initialized")
        print(f"   📦 Products: {n_products}")
        print(f"   🔁 Intervals: {n_intervals} (length {self.interval_length} rounds)")
        print(f"   💰 Prices: {prices}")
        print(f"   🏭 Inventory: {production_capacity}")

    def _generate_interval_distributions(self):
        dists = []
        for _ in range(self.n_intervals):
            interval_dists = {}
            for pid in range(self.n_products):
                low = self.rng.uniform(0.0, 0.5)
                high = self.rng.uniform(0.5, 1.0)
                if low > high:
                    low, high = high, low
                interval_dists[pid] = (low, high)
            dists.append(interval_dists)
        return dists

    def reset(self) -> Dict[str, Any]:
        self.current_round = 0
        self.remaining_inventory = self.production_capacity
        return {
            "round": self.current_round,
            "inventory": self.remaining_inventory
        }

    def _generate_buyer(self) -> MultiProductBuyer:
        interval_index = min(self.current_round // self.interval_length, self.n_intervals - 1)
        current_dists = self.valuation_distributions[interval_index]

        valuations = {
            pid: float(self.rng.uniform(low, high))
            for pid, (low, high) in current_dists.items()
        }
        return MultiProductBuyer(valuations)

    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict[str, Any], Dict[int, float], bool]:
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
        if not hasattr(self, "current_buyer") or self.current_buyer is None:
            return {}
        return self.current_buyer.valuations


def demo_piecewise_environment():
    env = MultiProductPiecewiseStationaryEnvironment(
        n_products=3,
        prices=[0.2, 0.4, 0.6, 0.8],
        production_capacity=20,
        total_rounds=15,
        n_intervals=3,
        random_seed=123
    )

    env.reset()

    for t in range(15):
        selected_prices = {0: 0.4, 1: 0.5, 2: 0.6}
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
    demo_piecewise_environment()
