# environments/stochastic.py
"""
Stochastic Environment for Single Product Pricing

This module implements a stochastic environment where buyer valuations
are drawn from a probability distribution for a single product type.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import BaseEnvironment, Buyer


class SingleProductStochasticEnvironment(BaseEnvironment):
    """
    Stochastic environment for single product pricing
    
    In each round, a buyer arrives with a valuation drawn from a probability distribution.
    The company sets a price and receives revenue if the buyer's valuation exceeds the price.
    """
    
    def __init__(self, 
                 prices: List[float],
                 production_capacity: int,
                 total_rounds: int = 1000,
                 valuation_distribution: str = "uniform",
                 valuation_params: Dict[str, float] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the stochastic environment
        
        Args:
            prices: List of possible prices (discrete set P)
            production_capacity: Maximum number of products that can be produced (B)
            total_rounds: Total number of rounds to simulate (T)
            valuation_distribution: Type of distribution for buyer valuations
                                    Options: "uniform", "normal", "beta", "exponential"
            valuation_params: Parameters for the chosen distribution
            random_seed: Random seed for reproducibility
        """
        # Single product environment (N=1, product_id=0)
        super().__init__(n_products=1, prices=prices, production_capacity=production_capacity)
        
        self.total_rounds = total_rounds
        self.valuation_distribution = valuation_distribution
        self.random_seed = random_seed
        
        # Set default parameters if none provided
        if valuation_params is None:
            valuation_params = self._get_default_params(valuation_distribution)
        self.valuation_params = valuation_params
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Track environment state
        self.current_buyer = None
        self.round_history = []
        
        print(f"🏭 Single Product Stochastic Environment initialized:")
        print(f"   Products: {self.n_products} (single product)")
        print(f"   Price range: {min(prices):.2f} - {max(prices):.2f}")
        print(f"   Production capacity: {production_capacity}")
        print(f"   Valuation distribution: {valuation_distribution}")
        print(f"   Distribution params: {valuation_params}")
        print(f"   Total rounds: {total_rounds}")
    
    def _get_default_params(self, distribution: str) -> Dict[str, float]:
        """Get default parameters for different distributions"""
        defaults = {
            "uniform": {"low": 0.0, "high": 1.0},
            "normal": {"mean": 0.5, "std": 0.2},
            "beta": {"alpha": 2.0, "beta": 2.0},
            "exponential": {"scale": 0.5}
        }
        return defaults.get(distribution, defaults["uniform"])
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state
        
        Returns:
            Initial state information
        """
        self.current_round = 0
        self.current_buyer = None
        self.round_history = []
        
        return {
            "round": self.current_round,
            "total_rounds": self.total_rounds,
            "production_capacity": self.production_capacity,
            "available_prices": self.prices.tolist()
        }
    
    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """
        Execute one round of the pricing game
        
        Args:
            selected_prices: Dict mapping product_id -> price
                           For single product: {0: price}
            
        Returns:
            Tuple of (buyer_info, rewards, done)
            - buyer_info: Information about the buyer's valuations and purchases
            - rewards: Dict mapping product_id -> reward received  
            - done: Whether the episode is finished
        """
        # Validate input
        if 0 not in selected_prices:
            raise ValueError("Single product environment expects price for product_id=0")
        
        price = selected_prices[0]
        if price not in self.prices:
            raise ValueError(f"Price {price} not in available price set {self.prices}")
        
        # Generate buyer for this round
        self.current_buyer = self._generate_buyer()
        buyer_valuation = self.current_buyer.get_valuation(0)
        
        # Determine if buyer makes purchase
        purchase_decision = self.current_buyer.make_purchases(selected_prices)
        bought = purchase_decision[0]  # Boolean: did buyer purchase product 0?
        
        # Calculate reward (revenue)
        # Revenue = price if buyer purchases, 0 otherwise
        reward = price if bought else 0.0
        
        # Store round information
        round_info = {
            "round": self.current_round,
            "buyer_valuation": buyer_valuation,
            "price_offered": price,
            "purchased": bought,
            "revenue": reward
        }
        self.round_history.append(round_info)
        
        # Prepare return values
        buyer_info = {
            "valuations": {0: buyer_valuation},
            "purchases": purchase_decision,
            "round": self.current_round
        }
        
        rewards = {0: reward}
        
        # Check if episode is done
        self.current_round += 1
        done = self.current_round >= self.total_rounds
        
        return buyer_info, rewards, done
    
    def get_buyer_valuations(self) -> Dict[int, float]:
        """
        Get current buyer's valuations
        
        Returns:
            Dict mapping product_id -> buyer's valuation for that product
        """
        if self.current_buyer is None:
            return {}
        return {0: self.current_buyer.get_valuation(0)}
    
    def _generate_buyer(self) -> Buyer:
        """
        Generate a new buyer with valuation drawn from the specified distribution
        
        Returns:
            Buyer object with randomly generated valuation
        """
        valuation = self._sample_valuation()
        
        # Ensure valuation is within reasonable bounds [0, max_price]
        valuation = max(0.0, min(valuation, max(self.prices)))
        
        return Buyer(valuations={0: valuation})
    
    def _sample_valuation(self) -> float:
        """
        Sample a valuation from the specified probability distribution
        
        Returns:
            Sampled valuation
        """
        if self.valuation_distribution == "uniform":
            return self.rng.uniform(
                self.valuation_params["low"], 
                self.valuation_params["high"]
            )
        
        elif self.valuation_distribution == "normal":
            # Sample from normal and clip to positive values
            val = self.rng.normal(
                self.valuation_params["mean"],
                self.valuation_params["std"]
            )
            return max(0.0, val)
        
        elif self.valuation_distribution == "beta":
            # Beta distribution naturally bounded between 0 and 1
            return self.rng.beta(
                self.valuation_params["alpha"],
                self.valuation_params["beta"]
            )
        
        elif self.valuation_distribution == "exponential":
            return self.rng.exponential(self.valuation_params["scale"])
        
        else:
            # Default to uniform if unknown distribution
            return self.rng.uniform(0.0, 1.0)
    
    def get_optimal_price(self) -> float:
        """
        Calculate the optimal price for this environment (oracle)
        
        This is the price that maximizes expected revenue given the valuation distribution.
        For most distributions, this requires numerical optimization.
        
        Returns:
            Optimal price
        """
        # For uniform distribution, we can calculate analytically
        if self.valuation_distribution == "uniform":
            low, high = self.valuation_params["low"], self.valuation_params["high"]
            # Optimal price for uniform distribution is (low + high) / 2
            optimal = (low + high) / 2
            
            # Find closest available price
            return self.prices[np.argmin(np.abs(self.prices - optimal))]
        
        # For other distributions, use numerical optimization over available prices
        best_price = self.prices[0]
        best_expected_revenue = 0.0
        
        for price in self.prices:
            expected_revenue = self._calculate_expected_revenue(price)
            if expected_revenue > best_expected_revenue:
                best_expected_revenue = expected_revenue
                best_price = price
        
        return best_price
    
    def _calculate_expected_revenue(self, price: float, n_samples: int = 10000) -> float:
        """
        Calculate expected revenue for a given price using Monte Carlo sampling
        
        Args:
            price: Price to evaluate
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Expected revenue
        """
        revenues = []
        
        for _ in range(n_samples):
            valuation = self._sample_valuation()
            # Revenue is price if valuation >= price, 0 otherwise
            revenue = price if valuation >= price else 0.0
            revenues.append(revenue)
        
        return np.mean(revenues)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the environment and its history
        
        Returns:
            Dictionary with various statistics
        """
        if not self.round_history:
            return {"message": "No rounds played yet"}
        
        valuations = [r["buyer_valuation"] for r in self.round_history]
        prices = [r["price_offered"] for r in self.round_history]
        purchases = [r["purchased"] for r in self.round_history]
        revenues = [r["revenue"] for r in self.round_history]
        
        return {
            "rounds_played": len(self.round_history),
            "total_revenue": sum(revenues),
            "average_revenue": np.mean(revenues),
            "purchase_rate": np.mean(purchases),
            "average_valuation": np.mean(valuations),
            "average_price": np.mean(prices),
            "valuation_std": np.std(valuations),
            "revenue_std": np.std(revenues)
        }
    
    def simulate_oracle(self, n_rounds: Optional[int] = None) -> List[float]:
        """
        Simulate optimal pricing strategy (oracle) for comparison
        
        Args:
            n_rounds: Number of rounds to simulate (default: total_rounds)
            
        Returns:
            List of rewards for each round under optimal strategy
        """
        if n_rounds is None:
            n_rounds = self.total_rounds
        
        optimal_price = self.get_optimal_price()
        oracle_rewards = []
        
        # Save current state
        current_round_backup = self.current_round
        
        for round_num in range(n_rounds):
            buyer = self._generate_buyer()
            valuation = buyer.get_valuation(0)
            
            # Oracle always uses optimal price
            purchase = valuation >= optimal_price
            reward = optimal_price if purchase else 0.0
            oracle_rewards.append(reward)
        
        # Restore state
        self.current_round = current_round_backup
        
        return oracle_rewards


def create_default_environment() -> SingleProductStochasticEnvironment:
    """
    Create a default single product stochastic environment for testing
    
    Returns:
        Configured environment instance
    """
    # Standard discrete price set
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return SingleProductStochasticEnvironment(
        prices=prices,
        production_capacity=10,  # Generous capacity for single product
        total_rounds=1000,
        valuation_distribution="uniform",
        valuation_params={"low": 0.0, "high": 1.0},
        random_seed=42
    )


def demo_environment():
    """
    Demonstrate the stochastic environment
    """
    print("🎮 Demo: Single Product Stochastic Environment")
    print("=" * 50)
    
    # Create environment
    env = create_default_environment()
    
    # Reset environment
    initial_state = env.reset()
    print(f"\n Initial state: {initial_state}")
    
    # Run a few rounds
    print(f"\n Running 5 demo rounds:")
    
    for round_num in range(5):
        # Random price selection for demo
        price = np.random.choice(env.prices)
        selected_prices = {0: price}
        
        buyer_info, rewards, done = env.step(selected_prices)
        
        print(f"Round {round_num + 1}:")
        print(f"  Price offered: {price:.2f}")
        print(f"  Buyer valuation: {buyer_info['valuations'][0]:.3f}")
        print(f"  Purchased: {buyer_info['purchases'][0]}")
        print(f"  Revenue: {rewards[0]:.2f}")
        print()
    
    # Show statistics
    stats = env.get_statistics()
    print("Environment Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Show optimal price
    optimal_price = env.get_optimal_price()
    print(f"\n Optimal price (oracle): {optimal_price:.2f}")


if __name__ == "__main__":
    demo_environment()