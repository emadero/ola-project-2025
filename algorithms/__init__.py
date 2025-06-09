"""
Base classes and interfaces for all pricing algorithms
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any


class BaseAlgorithm(ABC):
    """
    Base class for all pricing algorithms
    
    This defines the common interface that all team members must follow
    """
    
    def __init__(self, n_products: int, prices: List[float], production_capacity: int):
        """
        Initialize algorithm
        
        Args:
            n_products: Number of product types
            prices: List of possible prices (discrete set P)
            production_capacity: Maximum number of products that can be produced (B)
        """
        self.n_products = n_products
        self.prices = np.array(prices)
        self.production_capacity = production_capacity
        self.round = 0
        
        # Track history for analysis
        self.price_history = []
        self.reward_history = []
        self.regret_history = []
        
    @abstractmethod
    def select_prices(self) -> Dict[int, float]:
        """
        Select prices for all products in current round
        
        Returns:
            Dict mapping product_id -> selected_price
        """
        pass
    
    @abstractmethod
    def update(self, prices: Dict[int, float], rewards: Dict[int, float], 
               buyer_info: Dict[str, Any]) -> None:
        """
        Update algorithm with feedback from current round
        
        Args:
            prices: The prices that were selected
            rewards: The rewards received for each product
            buyer_info: Information about buyer behavior
        """
        self.round += 1
        
        # Store history
        self.price_history.append(prices.copy())
        self.reward_history.append(rewards.copy())
        
    def reset(self) -> None:
        """Reset algorithm to initial state"""
        self.round = 0
        self.price_history = []
        self.reward_history = []
        self.regret_history = []
    
    def get_cumulative_reward(self) -> float:
        """Calculate total cumulative reward"""
        total_reward = 0.0
        for round_rewards in self.reward_history:
            total_reward += sum(round_rewards.values())
        return total_reward
    
    def get_average_price(self, product_id: int) -> float:
        """Get average price selected for a specific product"""
        if not self.price_history:
            return 0.0
        
        prices = [round_prices.get(product_id, 0.0) for round_prices in self.price_history]
        return np.mean(prices)


class UCBMixin:
    """
    Mixin class providing common UCB functionality
    """
    
    def __init__(self):
        self.counts = {}  # Number of times each price was selected
        self.values = {}  # Average reward for each price
        
    def update_ucb_stats(self, product_id: int, price: float, reward: float):
        """Update UCB statistics for a product-price combination"""
        key = (product_id, price)
        
        if key not in self.counts:
            self.counts[key] = 0
            self.values[key] = 0.0
        
        self.counts[key] += 1
        # Update average reward using incremental formula
        self.values[key] += (reward - self.values[key]) / self.counts[key]
    
    def calculate_ucb_score(self, product_id: int, price: float, total_rounds: int, 
                           confidence_width: float = 2.0) -> float:
        """
        Calculate UCB score for a product-price combination
        
        Args:
            product_id: Product identifier
            price: Price to evaluate
            total_rounds: Total number of rounds played
            confidence_width: Width of confidence interval (usually sqrt(2))
            
        Returns:
            UCB score (mean + confidence interval)
        """
        key = (product_id, price)
        
        if key not in self.counts or self.counts[key] == 0:
            return float('inf')  # Unexplored actions get highest priority
        
        # UCB1 formula: mean + sqrt(2 * log(t) / n)
        mean_reward = self.values[key]
        
        if total_rounds <= 1:
            return mean_reward
            
        confidence_interval = confidence_width * np.sqrt(
            np.log(total_rounds) / self.counts[key]
        )
        
        return mean_reward + confidence_interval
    
    def get_price_statistics(self, product_id: int) -> Dict[str, Any]:
        """Get statistics for all prices of a product"""
        stats = {}
        for price in self.prices:
            key = (product_id, price)
            if key in self.counts:
                stats[price] = {
                    'count': self.counts[key],
                    'average_reward': self.values[key]
                }
            else:
                stats[price] = {
                    'count': 0,
                    'average_reward': 0.0
                }
        return stats


class ConstraintHandler:
    """
    Helper class for handling production constraints
    """
    
    @staticmethod
    def enforce_capacity_constraint(selected_products: Dict[int, bool], 
                                  capacity: int) -> Dict[int, bool]:
        """
        Enforce production capacity constraint
        
        Args:
            selected_products: Dict mapping product_id -> whether to produce
            capacity: Maximum number of products that can be produced
            
        Returns:
            Modified selection respecting capacity constraint
        """
        # Count total products selected
        total_selected = sum(selected_products.values())
        
        if total_selected <= capacity:
            return selected_products
        
        # If over capacity, prioritize somehow (this is a simple implementation)
        # More sophisticated algorithms might use different prioritization
        constrained_selection = selected_products.copy()
        selected_count = 0
        
        for product_id in sorted(selected_products.keys()):
            if selected_products[product_id] and selected_count < capacity:
                constrained_selection[product_id] = True
                selected_count += 1
            else:
                constrained_selection[product_id] = False
                
        return constrained_selection
    
    @staticmethod
    def calculate_production_cost(n_products: int, capacity: int) -> float:
        """
        Calculate cost of not using full production capacity
        
        This is a simple placeholder - teams can implement more sophisticated cost models
        """
        if n_products > capacity:
            return float('inf')  # Impossible to produce
        
        # Simple linear cost for unused capacity
        unused_capacity = capacity - n_products
        return 0.1 * unused_capacity  # Small penalty for unused capacity


# Make classes available for import
__all__ = ['BaseAlgorithm', 'UCBMixin', 'ConstraintHandler']