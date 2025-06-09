"""
Base interfaces and common classes for all environments
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any


class BaseEnvironment(ABC):
    """
    Base class for all pricing environments
    
    This defines the common interface that all team members must follow
    """
    
    def __init__(self, n_products: int, prices: List[float], production_capacity: int):
        """
        Initialize environment
        
        Args:
            n_products: Number of product types
            prices: List of possible prices (discrete set P)
            production_capacity: Maximum number of products that can be produced (B)
        """
        self.n_products = n_products
        self.prices = np.array(prices)
        self.production_capacity = production_capacity
        self.current_round = 0
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state
        
        Returns:
            Initial state information
        """
        self.current_round = 0
        pass
    
    @abstractmethod
    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """
        Execute one round of the pricing game
        
        Args:
            selected_prices: Dict mapping product_id -> price
            
        Returns:
            Tuple of (buyer_info, rewards, done)
            - buyer_info: Information about the buyer's valuations and purchases
            - rewards: Dict mapping product_id -> reward received
            - done: Whether the episode is finished
        """
        pass
    
    @abstractmethod
    def get_buyer_valuations(self) -> Dict[int, float]:
        """
        Generate buyer valuations for current round
        
        Returns:
            Dict mapping product_id -> buyer's valuation for that product
        """
        pass
    
    def get_available_prices(self) -> np.ndarray:
        """Get the discrete set of available prices"""
        return self.prices.copy()
    
    def get_production_capacity(self) -> int:
        """Get the production capacity constraint"""
        return self.production_capacity


class Buyer:
    """
    Represents a buyer with valuations for different products
    """
    
    def __init__(self, valuations: Dict[int, float]):
        """
        Initialize buyer
        
        Args:
            valuations: Dict mapping product_id -> valuation
        """
        self.valuations = valuations
    
    def make_purchases(self, prices: Dict[int, float]) -> Dict[int, bool]:
        """
        Determine which products the buyer will purchase
        
        Args:
            prices: Dict mapping product_id -> price
            
        Returns:
            Dict mapping product_id -> whether buyer purchases (True/False)
        """
        purchases = {}
        for product_id, price in prices.items():
            if product_id in self.valuations:
                # Buyer purchases if valuation >= price
                purchases[product_id] = self.valuations[product_id] >= price
            else:
                purchases[product_id] = False
        
        return purchases
    
    def get_valuation(self, product_id: int) -> float:
        """Get buyer's valuation for a specific product"""
        return self.valuations.get(product_id, 0.0)
    
    def get_all_valuations(self) -> Dict[int, float]:
        """Get all buyer valuations"""
        return self.valuations.copy()


# Make classes available for import
__all__ = ['BaseEnvironment', 'Buyer']