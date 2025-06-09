# algorithms/single_product/ucb.py
"""
UCB1 Algorithm for Single Product Pricing (without inventory constraints)
Assigned to: Federico Madero

This module implements the Upper Confidence Bound (UCB1) algorithm for pricing
a single product in a stochastic environment, ignoring inventory constraints.
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms import BaseAlgorithm, UCBMixin


class UCB1Algorithm(BaseAlgorithm, UCBMixin):
    """
    UCB1 algorithm for single product pricing without inventory constraints
    
    The algorithm maintains confidence bounds for each price and selects
    the price with the highest upper confidence bound.
    
    UCB1 Formula: UCB(price) = mean_reward(price) + sqrt(2 * log(t) / n(price))
    where:
    - mean_reward(price): average reward received for this price
    - t: total number of rounds
    - n(price): number of times this price was selected
    """
    
    def __init__(self, 
                 prices: List[float], 
                 production_capacity: int,
                 confidence_width: float = np.sqrt(2),
                 random_seed: Optional[int] = None):
        """
        Initialize UCB1 algorithm
        
        Args:
            prices: List of possible prices (discrete set P)
            production_capacity: Maximum production capacity (ignored in this version)
            confidence_width: Width parameter for confidence intervals (default: sqrt(2))
            random_seed: Random seed for reproducibility
        """
        # Initialize base classes
        BaseAlgorithm.__init__(self, n_products=1, prices=prices, production_capacity=production_capacity)
        UCBMixin.__init__(self)
        
        self.confidence_width = confidence_width
        self.rng = np.random.RandomState(random_seed)
        
        # Single product ID
        self.product_id = 0
        
        # Track performance
        self.exploration_count = 0  # Number of times we selected unexplored prices
        self.exploitation_count = 0  # Number of times we exploited best known price
        
        print(f"🎯 UCB1 Algorithm initialized:")
        print(f"   📦 Product: Single product (ID: {self.product_id})")
        print(f"   💰 Available prices: {len(prices)} ({min(prices):.2f} - {max(prices):.2f})")
        print(f"   🔍 Confidence width: {confidence_width:.3f}")
        print(f"   ⚠️  Inventory constraints: IGNORED")
    
    def select_prices(self) -> Dict[int, float]:
        """
        Select price for the single product using UCB1 strategy
        
        Returns:
            Dict mapping product_id -> selected_price
        """
        # For single product, we only select price for product_id=0
        selected_price = self._select_price_ucb1()
        
        return {self.product_id: selected_price}
    
    def _select_price_ucb1(self) -> float:
        """
        Select price using UCB1 algorithm
        
        Returns:
            Selected price
        """
        best_price = None
        best_ucb_score = -float('inf')
        
        # Calculate UCB score for each price
        for price in self.prices:
            ucb_score = self.calculate_ucb_score(
                self.product_id, 
                price, 
                max(1, self.round),  # Avoid log(0)
                self.confidence_width
            )
            
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_price = price
        
        # Track exploration vs exploitation
        key = (self.product_id, best_price)
        if key not in self.counts or self.counts[key] == 0:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1
        
        return best_price
    
    def update(self, prices: Dict[int, float], rewards: Dict[int, float], 
               buyer_info: Dict[str, Any]) -> None:
        """
        Update algorithm with feedback from current round
        
        Args:
            prices: The prices that were selected
            rewards: The rewards received for each product
            buyer_info: Information about buyer behavior (valuations, purchases)
        """
        # Call parent update
        super().update(prices, rewards, buyer_info)
        
        # Update UCB statistics for single product
        if self.product_id in prices and self.product_id in rewards:
            price = prices[self.product_id]
            reward = rewards[self.product_id]
            
            self.update_ucb_stats(self.product_id, price, reward)
    
    def get_best_price(self) -> float:
        """
        Get the price with highest average reward (exploitation only)
        
        Returns:
            Price with highest average reward
        """
        best_price = self.prices[0]
        best_reward = -float('inf')
        
        for price in self.prices:
            key = (self.product_id, price)
            if key in self.values and self.values[key] > best_reward:
                best_reward = self.values[key]
                best_price = price
        
        return best_price
    
    def get_ucb_scores(self) -> Dict[float, float]:
        """
        Get current UCB scores for all prices
        
        Returns:
            Dict mapping price -> UCB score
        """
        scores = {}
        for price in self.prices:
            scores[price] = self.calculate_ucb_score(
                self.product_id,
                price,
                max(1, self.round),
                self.confidence_width
            )
        return scores
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm statistics
        
        Returns:
            Dictionary with algorithm performance statistics
        """
        price_stats = self.get_price_statistics(self.product_id)
        
        # Calculate exploration vs exploitation ratio
        total_actions = self.exploration_count + self.exploitation_count
        exploration_ratio = self.exploration_count / max(1, total_actions)
        
        # Find most and least selected prices
        most_selected_price = None
        least_selected_price = None
        max_count = -1
        min_count = float('inf')
        
        for price, stats in price_stats.items():
            if stats['count'] > max_count:
                max_count = stats['count']
                most_selected_price = price
            if stats['count'] < min_count and stats['count'] > 0:
                min_count = stats['count']
                least_selected_price = price
        
        return {
            "algorithm": "UCB1",
            "total_rounds": self.round,
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "exploration_ratio": exploration_ratio,
            "best_price": self.get_best_price(),
            "most_selected_price": most_selected_price,
            "most_selected_count": max_count,
            "least_selected_price": least_selected_price,
            "least_selected_count": min_count,
            "confidence_width": self.confidence_width,
            "total_reward": self.get_cumulative_reward(),
            "average_reward": self.get_cumulative_reward() / max(1, self.round)
        }
    
    def get_regret_bounds(self) -> Dict[str, float]:
        """
        Calculate theoretical regret bounds for UCB1
        
        Returns:
            Dictionary with regret bound information
        """
        if self.round == 0:
            return {"theoretical_regret_bound": 0.0}
        
        # UCB1 theoretical regret bound: O(sqrt(K * log(T) * T))
        # where K is number of arms (prices) and T is number of rounds
        K = len(self.prices)
        T = self.round
        
        if T > 1:
            # Simplified bound calculation
            theoretical_bound = np.sqrt(K * np.log(T) * T)
        else:
            theoretical_bound = 0.0
        
        return {
            "theoretical_regret_bound": theoretical_bound,
            "regret_bound_per_round": theoretical_bound / max(1, T),
            "number_of_arms": K,
            "rounds_played": T
        }


def create_default_ucb1() -> UCB1Algorithm:
    """
    Create a default UCB1 algorithm for testing
    
    Returns:
        Configured UCB1 algorithm instance
    """
    # Standard discrete price set
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return UCB1Algorithm(
        prices=prices,
        production_capacity=10,  # Will be ignored
        confidence_width=np.sqrt(2),
        random_seed=42
    )


def demo_ucb1():
    """
    Demonstrate the UCB1 algorithm
    """
    print("🎮 Demo: UCB1 Algorithm for Single Product Pricing")
    print("=" * 60)
    
    # Create algorithm
    ucb1 = create_default_ucb1()
    
    # Simulate some rounds
    print(f"\n🎯 Running 10 demo rounds:")
    
    # Mock buyer responses (normally this would come from environment)
    mock_buyer_valuations = [0.7, 0.3, 0.8, 0.5, 0.9, 0.2, 0.6, 0.4, 0.75, 0.35]
    
    for round_num in range(10):
        # Select price
        selected_prices = ucb1.select_prices()
        price = selected_prices[0]
        
        # Mock buyer response
        buyer_valuation = mock_buyer_valuations[round_num]
        purchased = buyer_valuation >= price
        reward = price if purchased else 0.0
        
        # Mock buyer info
        buyer_info = {
            "valuations": {0: buyer_valuation},
            "purchases": {0: purchased},
            "round": round_num
        }
        
        rewards = {0: reward}
        
        # Update algorithm
        ucb1.update(selected_prices, rewards, buyer_info)
        
        print(f"Round {round_num + 1}:")
        print(f"  💰 Price selected: {price:.2f}")
        print(f"  👤 Buyer valuation: {buyer_valuation:.3f}")
        print(f"  🛒 Purchased: {purchased}")
        print(f"  💵 Revenue: {reward:.2f}")
    
    # Show UCB scores
    print(f"\n📊 Final UCB Scores:")
    ucb_scores = ucb1.get_ucb_scores()
    for price, score in sorted(ucb_scores.items()):
        if score == float('inf'):
            print(f"  Price {price:.2f}: UNEXPLORED")
        else:
            print(f"  Price {price:.2f}: {score:.3f}")
    
    # Show algorithm statistics
    stats = ucb1.get_algorithm_stats()
    print(f"\n📈 Algorithm Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Show regret bounds
    bounds = ucb1.get_regret_bounds()
    print(f"\n🎯 Regret Analysis:")
    for key, value in bounds.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_ucb1()
