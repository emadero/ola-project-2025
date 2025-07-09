"""
UCB1 Algorithm for Single Product Pricing (Multi-Armed Bandit Approach)

This module implements UCB1 following the correct Multi-Armed Bandit paradigm:
- Each price is an "arm" with unknown expected reward
- UCB1 balances exploration vs exploitation across price arms
- Follows the UCB-like approach from auction theory
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms import BaseAlgorithm, UCBMixin


class UCB1PricingAlgorithm(BaseAlgorithm, UCBMixin):
    """
    UCB1 algorithm for single product pricing following Multi-Armed Bandit paradigm
    
    Key insight: Each price is a separate "arm" in the bandit problem
    - Arms: Available prices [p1, p2, ..., pK]  
    - Action: Select one price (arm) per round
    - Reward: Revenue from that price
    - Goal: Learn which prices give best expected revenue
    
    UCB1 Formula per arm: μ̂(p) + sqrt(2 * log(t) / n(p))
    where:
    - μ̂(p): estimated mean reward for price p
    - t: total rounds played
    - n(p): times price p was selected
    """
    
    def __init__(self, 
                 prices: List[float], 
                 production_capacity: int,
                 confidence_width: float = np.sqrt(2),
                 random_seed: Optional[int] = None):
        """
        Initialize UCB1 pricing algorithm
        
        Args:
            prices: List of possible prices (these are our "arms")
            production_capacity: Not used in basic UCB1 (ignored)
            confidence_width: UCB confidence width (usually sqrt(2))
            random_seed: Random seed for reproducibility
        """
        # Initialize base classes
        BaseAlgorithm.__init__(self, n_products=1, prices=prices, production_capacity=production_capacity)
        UCBMixin.__init__(self)
        
        self.confidence_width = confidence_width
        self.rng = np.random.RandomState(random_seed)
        
        # Single product ID
        self.product_id = 0
        
        # Multi-Armed Bandit statistics
        # Each price is an arm with its own statistics
        self.arm_counts = {}     # n(p): times each price was selected
        self.arm_rewards = {}    # sum of rewards for each price
        self.arm_means = {}      # μ̂(p): estimated mean reward for each price
        
        # Initialize arms (prices)
        for price in self.prices:
            arm_id = self._price_to_arm_id(price)
            self.arm_counts[arm_id] = 0
            self.arm_rewards[arm_id] = 0.0
            self.arm_means[arm_id] = 0.0
        
        # Track exploration vs exploitation
        self.exploration_count = 0
        self.exploitation_count = 0
        
        print(f"UCB1 Pricing Algorithm initialized:")
        print(f"  Product: Single product (Multi-Armed Bandit)")
        print(f"  Arms (prices): {len(prices)} arms from {min(prices):.2f} to {max(prices):.2f}")
        print(f"  Confidence width: {confidence_width:.3f}")
        print(f"  Inventory constraints: IGNORED")
        print(f"  Arms: {[f'{p:.2f}' for p in prices]}")
    
    def _price_to_arm_id(self, price: float) -> int:
        """Convert price to arm ID (index in price array)"""
        return list(self.prices).index(price)
    
    def _arm_id_to_price(self, arm_id: int) -> float:
        """Convert arm ID to price"""
        return self.prices[arm_id]
    
    def select_prices(self) -> Dict[int, float]:
        """
        Select price using UCB1 Multi-Armed Bandit strategy
        
        For each arm (price), calculate:
        UCB(p) = μ̂(p) + sqrt(2 * log(t) / n(p))
        
        Select arm with highest UCB score.
        
        Returns:
            Dict mapping product_id -> selected_price
        """
        selected_arm = self._select_arm_ucb1()
        selected_price = self._arm_id_to_price(selected_arm)
        
        return {self.product_id: selected_price}
    
    def _select_arm_ucb1(self) -> int:
        """
        Select arm (price) using UCB1 algorithm
        
        Returns:
            Arm ID (index) of selected price
        """
        best_arm = None
        best_ucb_score = -float('inf')
        
        # Calculate UCB score for each arm (price)
        for arm_id in range(len(self.prices)):
            ucb_score = self._calculate_ucb_score(arm_id)
            
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_arm = arm_id
        
        # Track exploration vs exploitation
        if self.arm_counts[best_arm] == 0:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1
        
        return best_arm
    
    def _calculate_ucb_score(self, arm_id: int) -> float:
        """
        Calculate UCB1 score for a specific arm (price)
        
        UCB1 formula: μ̂(arm) + sqrt(2 * log(t) / n(arm))
        
        Args:
            arm_id: ID of the arm (price index)
            
        Returns:
            UCB score for this arm
        """
        # If arm never played, give infinite score (exploration)
        if self.arm_counts[arm_id] == 0:
            return float('inf')
        
        # If no rounds played yet, return mean estimate
        if self.round <= 0:
            return self.arm_means[arm_id]
        
        # Standard UCB1 formula
        mean_reward = self.arm_means[arm_id]
        confidence_radius = self.confidence_width * np.sqrt(
            np.log(self.round) / self.arm_counts[arm_id]
        )
        
        return mean_reward + confidence_radius
    
    def update(self, prices: Dict[int, float], rewards: Dict[int, float], 
               buyer_info: Dict[str, Any]) -> None:
        """
        Update Multi-Armed Bandit statistics
        
        Args:
            prices: The prices that were selected {product_id: price}
            rewards: The rewards received {product_id: reward}
            buyer_info: Information about buyer behavior
        """
        # Call parent update for history tracking
        super().update(prices, rewards, buyer_info)
        
        # Update Multi-Armed Bandit statistics
        if self.product_id in prices and self.product_id in rewards:
            selected_price = prices[self.product_id]
            received_reward = rewards[self.product_id]
            
            # Convert price to arm ID
            arm_id = self._price_to_arm_id(selected_price)
            
            # Update arm statistics
            self._update_arm_statistics(arm_id, received_reward)
    
    def _update_arm_statistics(self, arm_id: int, reward: float) -> None:
        """
        Update statistics for a specific arm after observing reward
        
        Args:
            arm_id: ID of the arm that was played
            reward: Reward received from playing this arm
        """
        # Increment count
        self.arm_counts[arm_id] += 1
        
        # Update total rewards
        self.arm_rewards[arm_id] += reward
        
        # Update mean reward (running average)
        self.arm_means[arm_id] = self.arm_rewards[arm_id] / self.arm_counts[arm_id]
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics for all arms (prices)
        
        Returns:
            Dictionary with statistics for each arm
        """
        arm_stats = {}
        
        for arm_id in range(len(self.prices)):
            price = self._arm_id_to_price(arm_id)
            
            arm_stats[f"price_{price:.2f}"] = {
                "arm_id": arm_id,
                "price": price,
                "times_selected": self.arm_counts[arm_id],
                "total_reward": self.arm_rewards[arm_id],
                "mean_reward": self.arm_means[arm_id],
                "ucb_score": self._calculate_ucb_score(arm_id) if self.round > 0 else float('inf')
            }
        
        return arm_stats
    
    def get_best_arm(self) -> int:
        """
        Get arm with highest empirical mean (pure exploitation)
        
        Returns:
            Arm ID of best arm based on current estimates
        """
        best_arm = 0
        best_mean = self.arm_means[0]
        
        for arm_id in range(len(self.prices)):
            if self.arm_means[arm_id] > best_mean:
                best_mean = self.arm_means[arm_id]
                best_arm = arm_id
        
        return best_arm
    
    def get_best_price(self) -> float:
        """
        Get price with highest empirical mean reward
        
        Returns:
            Best price based on current estimates
        """
        best_arm = self.get_best_arm()
        return self._arm_id_to_price(best_arm)
    
    def get_ucb_scores(self) -> Dict[float, float]:
        """
        Get current UCB scores for all arms (prices)
        
        Returns:
            Dict mapping price -> UCB score
        """
        scores = {}
        for arm_id in range(len(self.prices)):
            price = self._arm_id_to_price(arm_id)
            scores[price] = self._calculate_ucb_score(arm_id)
        return scores
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm statistics
        
        Returns:
            Dictionary with algorithm performance statistics
        """
        # Find most and least selected arms
        most_selected_arm = max(range(len(self.prices)), key=lambda x: self.arm_counts[x])
        least_selected_count = min(self.arm_counts.values())
        least_selected_arms = [i for i in range(len(self.prices)) if self.arm_counts[i] == least_selected_count]
        
        # Calculate exploration vs exploitation ratio
        total_actions = self.exploration_count + self.exploitation_count
        exploration_ratio = self.exploration_count / max(1, total_actions)
        
        return {
            "algorithm": "UCB1 Multi-Armed Bandit",
            "total_rounds": self.round,
            "num_arms": len(self.prices),
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "exploration_ratio": exploration_ratio,
            "best_arm_id": self.get_best_arm(),
            "best_price": self.get_best_price(),
            "most_selected_arm": most_selected_arm,
            "most_selected_price": self._arm_id_to_price(most_selected_arm),
            "most_selected_count": self.arm_counts[most_selected_arm],
            "least_selected_count": least_selected_count,
            "confidence_width": self.confidence_width,
            "total_reward": self.get_cumulative_reward(),
            "average_reward": self.get_cumulative_reward() / max(1, self.round)
        }
    
    def get_regret_bounds(self) -> Dict[str, float]:
        """
        Calculate theoretical regret bounds for UCB1
        
        UCB1 theoretical regret bound: O(sqrt(K * log(T) * T))
        where K = number of arms, T = number of rounds
        
        Returns:
            Dictionary with regret bound information
        """
        if self.round == 0:
            return {
                "theoretical_regret_bound": 0.0,
                "regret_bound_per_round": 0.0,
                "number_of_arms": len(self.prices),
                "rounds_played": 0,
                "log_factor": 0.0
            }
        
        K = len(self.prices)  # Number of arms
        T = self.round        # Number of rounds
        
        if T > 1:
            # UCB1 regret bound: 8 * sqrt(K * log(T) * T)
            # This is a simplified version of the theoretical bound
            theoretical_bound = 8 * np.sqrt(K * np.log(T) * T)
            log_factor = np.log(T)
        else:
            theoretical_bound = 0.0
            log_factor = 0.0
        
        return {
            "theoretical_regret_bound": theoretical_bound,
            "regret_bound_per_round": theoretical_bound / max(1, T),
            "number_of_arms": K,
            "rounds_played": T,
            "log_factor": log_factor
        }
    
    def simulate_oracle_regret(self, environment) -> float:
        """
        Calculate regret vs oracle (best fixed arm in hindsight)
        
        Args:
            environment: Environment with oracle functionality
            
        Returns:
            Cumulative regret vs oracle
        """
        if not hasattr(environment, 'get_optimal_price'):
            return 0.0
        
        # Get oracle performance
        oracle_price = environment.get_optimal_price()
        oracle_rewards = environment.simulate_oracle(n_rounds=self.round)
        oracle_total = sum(oracle_rewards)
        
        # Get our performance
        our_total = self.get_cumulative_reward()
        
        # Regret = Oracle - Ours
        regret = oracle_total - our_total
        return max(0.0, regret)  # Regret can't be negative


def create_default_ucb1() -> UCB1PricingAlgorithm:
    """
    Create a default UCB1 pricing algorithm for testing
    
    Returns:
        Configured UCB1 algorithm instance
    """
    # Standard discrete price set (these are our "arms")
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return UCB1PricingAlgorithm(
        prices=prices,
        production_capacity=10,  # Ignored in basic UCB1
        confidence_width=np.sqrt(2),  # Standard UCB1 parameter
        random_seed=42
    )
