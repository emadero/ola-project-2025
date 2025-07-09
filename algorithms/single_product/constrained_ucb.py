"""
UCB-like Algorithm for Single Product Pricing with Constraints

This module implements the UCB-like approach from auction theory adapted to pricing with constraints.
Follows the paradigm: "Extend the UCB-like approach that we saw for auctions to the pricing problem"

Key insight from auction theory:
- Estimate reward function f̄ with upper confidence bound f̄^UCB
- Estimate constraint function c̄ with lower confidence bound c̄^LCB  
- Solve: max f̄^UCB(a) subject to c̄^LCB(a) ≤ budget

Adapted to pricing:
- Arms: Each price is an arm
- Rewards: Revenue from each price
- Constraints: Capacity consumption per price
- Goal: Max revenue subject to capacity constraints
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms import BaseAlgorithm, UCBMixin
from algorithms.single_product.ucb import UCB1PricingAlgorithm


class UCBConstrainedPricingAlgorithm(UCB1PricingAlgorithm):
    """
    UCB-like algorithm for constrained pricing following auction theory paradigm
    
    Extension of Multi-Armed Bandit UCB1 to handle capacity constraints.
    
    At each round t:
    1. Estimate f̄^UCB_t(arm): Upper confidence bound for reward of each arm
    2. Estimate c̄^LCB_t(arm): Lower confidence bound for constraint cost of each arm  
    3. Solve: argmax_arm f̄^UCB_t(arm) subject to c̄^LCB_t(arm) ≤ remaining_capacity
    
    Where:
    - f̄(arm) = expected revenue from price arm
    - c̄(arm) = expected capacity consumption from price arm
    """
    
    def __init__(self, 
                 prices: List[float], 
                 production_capacity: int,
                 confidence_width: float = np.sqrt(2),
                 constraint_confidence_width: float = np.sqrt(2),
                 random_seed: Optional[int] = None):
        """
        Initialize UCB-like constrained pricing algorithm
        
        Args:
            prices: List of possible prices (arms)
            production_capacity: Maximum production capacity per round
            confidence_width: UCB confidence width for rewards
            constraint_confidence_width: Confidence width for constraint estimates
            random_seed: Random seed for reproducibility
        """
        # Initialize parent UCB1
        super().__init__(prices, production_capacity, confidence_width, random_seed)
        
        self.constraint_confidence_width = constraint_confidence_width
        
        # Constraint (capacity) statistics for each arm
        # c̄(arm) = expected capacity consumption for each price arm
        self.constraint_counts = {}    # Times each arm was played for constraint learning
        self.constraint_totals = {}    # Total constraint consumption per arm
        self.constraint_means = {}     # Mean constraint consumption per arm
        
        # Initialize constraint statistics for each arm (price)
        for arm_id in range(len(self.prices)):
            self.constraint_counts[arm_id] = 0
            self.constraint_totals[arm_id] = 0.0
            self.constraint_means[arm_id] = 0.0  # Initially assume no capacity needed
        
        # Capacity tracking
        self.total_capacity = production_capacity
        self.remaining_capacity = production_capacity
        self.capacity_used_per_round = []
        
        # Constraint violation tracking
        self.constraint_violations = 0
        self.feasible_arms_per_round = []
        
        print(f" UCB-Constrained Pricing Algorithm initialized:")
        print(f"   Product: Single product (Constrained Multi-Armed Bandit)")
        print(f"   Arms (prices): {len(prices)} arms from {min(prices):.2f} to {max(prices):.2f}")
        print(f"   Production capacity: {production_capacity}")
        print(f"   Reward confidence width: {confidence_width:.3f}")
        print(f"   Constraint confidence width: {constraint_confidence_width:.3f}")
        print(f"   Inventory constraints: ACTIVE (UCB-like approach)")
    
    def select_prices(self) -> Dict[int, float]:
        """
        Select price using UCB-like constrained optimization
        
        Solves: argmax_arm f̄^UCB(arm) subject to c̄^LCB(arm) ≤ remaining_capacity
        
        Returns:
            Dict mapping product_id -> selected_price (empty if infeasible)
        """
        # Check capacity constraint first
        if self.remaining_capacity <= 0:
            self.feasible_arms_per_round.append([])
            return {}  # No capacity available
        
        # Step 1: Calculate UCB bounds for all arms
        reward_ucb_bounds = self._calculate_reward_ucb_bounds()
        constraint_lcb_bounds = self._calculate_constraint_lcb_bounds()
        
        # Step 2: Find feasible arms (satisfy constraint)
        feasible_arms = self._find_feasible_arms(constraint_lcb_bounds)
        
        if not feasible_arms:
            # No feasible arms - cannot produce this round
            self.feasible_arms_per_round.append([])
            return {}
        
        # Step 3: Among feasible arms, select one with highest reward UCB
        best_arm = self._select_best_feasible_arm(feasible_arms, reward_ucb_bounds)
        selected_price = self._arm_id_to_price(best_arm)
        
        # Track feasible arms for analysis
        self.feasible_arms_per_round.append(feasible_arms.copy())
        
        return {self.product_id: selected_price}
    
    def _calculate_reward_ucb_bounds(self) -> Dict[int, float]:
        """
        Calculate f̄^UCB_t(arm) for all arms
        
        f̄^UCB_t(arm) = μ̂(arm) + sqrt(2 * log(t) / n(arm))
        
        Returns:
            Dict mapping arm_id -> upper confidence bound for reward
        """
        ucb_bounds = {}
        
        for arm_id in range(len(self.prices)):
            # Use parent method for reward UCB calculation
            ucb_bounds[arm_id] = self._calculate_ucb_score(arm_id)
        
        return ucb_bounds
    
    def _calculate_constraint_lcb_bounds(self) -> Dict[int, float]:
        """
        Calculate c̄^LCB_t(arm) for all arms
        
        c̄^LCB_t(arm) = ĉ(arm) - sqrt(2 * log(t) / n_c(arm))
        
        Lower confidence bound ensures we're conservative about constraint consumption.
        
        Returns:
            Dict mapping arm_id -> lower confidence bound for constraint cost
        """
        lcb_bounds = {}
        
        for arm_id in range(len(self.prices)):
            if self.constraint_counts[arm_id] == 0:
                # Unobserved arm - assume optimistic constraint cost
                # For pricing: each production uses exactly 1 capacity unit
                lcb_bounds[arm_id] = 0.5  # Optimistic estimate
            else:
                mean_constraint = self.constraint_means[arm_id]
                
                if self.round <= 1:
                    confidence_radius = 0.0
                else:
                    confidence_radius = self.constraint_confidence_width * np.sqrt(
                        np.log(self.round) / self.constraint_counts[arm_id]
                    )
                
                # Lower confidence bound (conservative estimate)
                lcb_bounds[arm_id] = max(0.0, mean_constraint - confidence_radius)
        
        return lcb_bounds
    
    def _find_feasible_arms(self, constraint_lcb_bounds: Dict[int, float]) -> List[int]:
        """
        Find arms that satisfy capacity constraint
        
        Feasible if: c̄^LCB(arm) ≤ remaining_capacity
        
        Args:
            constraint_lcb_bounds: Lower confidence bounds for constraint costs
            
        Returns:
            List of feasible arm IDs
        """
        feasible_arms = []
        
        for arm_id in range(len(self.prices)):
            constraint_lcb = constraint_lcb_bounds[arm_id]
            
            if constraint_lcb <= self.remaining_capacity:
                feasible_arms.append(arm_id)
        
        return feasible_arms
    
    def _select_best_feasible_arm(self, feasible_arms: List[int], 
                                 reward_ucb_bounds: Dict[int, float]) -> int:
        """
        Among feasible arms, select the one with highest reward UCB
        
        Args:
            feasible_arms: List of feasible arm IDs
            reward_ucb_bounds: Upper confidence bounds for rewards
            
        Returns:
            Arm ID with highest reward UCB among feasible arms
        """
        best_arm = feasible_arms[0]
        best_ucb = reward_ucb_bounds[best_arm]
        
        for arm_id in feasible_arms[1:]:
            if reward_ucb_bounds[arm_id] > best_ucb:
                best_ucb = reward_ucb_bounds[arm_id]
                best_arm = arm_id
        
        # Track exploration vs exploitation for feasible arms
        if self.arm_counts[best_arm] == 0:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1
        
        return best_arm
    
    def update(self, prices: Dict[int, float], rewards: Dict[int, float], 
               buyer_info: Dict[str, Any]) -> None:
        """
        Update both reward and constraint statistics
        
        Args:
            prices: Prices that were selected (empty if no production)
            rewards: Rewards received (empty if no production)
            buyer_info: Information about buyer behavior
        """
        # Update parent UCB1 statistics for rewards
        super().update(prices, rewards, buyer_info)
        
        # Update constraint statistics
        if prices:  # If we produced something
            selected_price = prices[self.product_id]
            arm_id = self._price_to_arm_id(selected_price)
            
            # For single product pricing: each production consumes exactly 1 capacity
            capacity_consumed = 1.0
            
            self._update_constraint_statistics(arm_id, capacity_consumed)
            
            # Update capacity tracking
            self.remaining_capacity -= capacity_consumed
            self.capacity_used_per_round.append(capacity_consumed)
        else:
            # No production this round
            self.capacity_used_per_round.append(0.0)
    
    def _update_constraint_statistics(self, arm_id: int, constraint_cost: float) -> None:
        """
        Update constraint statistics for a specific arm
        
        Args:
            arm_id: ID of the arm that was played
            constraint_cost: Capacity consumed by playing this arm
        """
        # Update constraint statistics
        self.constraint_counts[arm_id] += 1
        self.constraint_totals[arm_id] += constraint_cost
        self.constraint_means[arm_id] = self.constraint_totals[arm_id] / self.constraint_counts[arm_id]
    
    def reset_capacity(self) -> None:
        """Reset production capacity (e.g., at start of new period)"""
        self.remaining_capacity = self.total_capacity
        print(f" Capacity reset: {self.remaining_capacity}/{self.total_capacity} available")
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """
        Get detailed constraint statistics for all arms
        
        Returns:
            Dictionary with constraint statistics for each arm
        """
        constraint_stats = {}
        
        for arm_id in range(len(self.prices)):
            price = self._arm_id_to_price(arm_id)
            
            constraint_stats[f"price_{price:.2f}"] = {
                "arm_id": arm_id,
                "price": price,
                "constraint_observations": self.constraint_counts[arm_id],
                "total_constraint_cost": self.constraint_totals[arm_id],
                "mean_constraint_cost": self.constraint_means[arm_id],
                "constraint_lcb": self._calculate_constraint_lcb_bounds()[arm_id]
            }
        
        return constraint_stats
    
    def get_capacity_statistics(self) -> Dict[str, Any]:
        """
        Get capacity utilization and constraint statistics
        
        Returns:
            Dictionary with capacity-related metrics
        """
        total_rounds = len(self.capacity_used_per_round)
        total_capacity_used = sum(self.capacity_used_per_round)
        
        # Calculate feasibility statistics
        total_feasible_arms = sum(len(arms) for arms in self.feasible_arms_per_round)
        avg_feasible_arms = total_feasible_arms / max(1, len(self.feasible_arms_per_round))
        
        # Count rounds where no production was possible
        infeasible_rounds = sum(1 for arms in self.feasible_arms_per_round if len(arms) == 0)
        
        return {
            "total_capacity": self.total_capacity,
            "remaining_capacity": self.remaining_capacity,
            "capacity_used": total_capacity_used,
            "capacity_utilization": total_capacity_used / max(1, self.total_capacity),
            "production_rounds": sum(1 for x in self.capacity_used_per_round if x > 0),
            "infeasible_rounds": infeasible_rounds,
            "average_feasible_arms": avg_feasible_arms,
            "production_rate": sum(1 for x in self.capacity_used_per_round if x > 0) / max(1, total_rounds)
        }
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm statistics including constraints
        
        Returns:
            Dictionary with algorithm performance and constraint statistics
        """
        # Get base UCB1 statistics
        base_stats = super().get_algorithm_stats()
        
        # Add constraint-specific statistics
        capacity_stats = self.get_capacity_statistics()
        constraint_info = {
            "algorithm": "UCB-Constrained Pricing (Auction-like)",
            "constraint_confidence_width": self.constraint_confidence_width,
            **capacity_stats
        }
        
        # Merge statistics
        return {**base_stats, **constraint_info}
    
    def get_feasibility_analysis(self) -> Dict[str, Any]:
        """
        Analyze feasibility patterns over time
        
        Returns:
            Dictionary with feasibility analysis
        """
        if not self.feasible_arms_per_round:
            return {
                "total_rounds": 0,
                "rounds_with_feasible_arms": 0,
                "rounds_infeasible": 0,
                "average_feasible_arms": 0.0,
                "min_feasible_arms": 0,
                "max_feasible_arms": 0,
                "feasibility_rate": 0.0
            }
        
        # Analyze feasibility trends
        feasible_counts = [len(arms) for arms in self.feasible_arms_per_round]
        
        return {
            "total_rounds": len(feasible_counts),
            "rounds_with_feasible_arms": sum(1 for count in feasible_counts if count > 0),
            "rounds_infeasible": sum(1 for count in feasible_counts if count == 0),
            "average_feasible_arms": np.mean(feasible_counts) if feasible_counts else 0.0,
            "min_feasible_arms": min(feasible_counts) if feasible_counts else 0,
            "max_feasible_arms": max(feasible_counts) if feasible_counts else 0,
            "feasibility_rate": sum(1 for count in feasible_counts if count > 0) / len(feasible_counts) if feasible_counts else 0.0
        }
    
    def compare_ucb_vs_constraint_bounds(self) -> Dict[str, Any]:
        """
        Compare reward UCB bounds vs constraint LCB bounds for analysis
        
        Returns:
            Dictionary comparing bounds for all arms
        """
        reward_bounds = self._calculate_reward_ucb_bounds()
        constraint_bounds = self._calculate_constraint_lcb_bounds()
        
        comparison = {}
        for arm_id in range(len(self.prices)):
            price = self._arm_id_to_price(arm_id)
            
            comparison[f"price_{price:.2f}"] = {
                "price": price,
                "reward_ucb": reward_bounds[arm_id],
                "constraint_lcb": constraint_bounds[arm_id],
                "is_feasible": constraint_bounds[arm_id] <= self.remaining_capacity,
                "reward_mean": self.arm_means[arm_id],
                "constraint_mean": self.constraint_means[arm_id]
            }
        
        return comparison


def create_default_constrained_ucb1(capacity: int = 10) -> UCBConstrainedPricingAlgorithm:
    """
    Create a default UCB-Constrained pricing algorithm for testing
    
    Args:
        capacity: Production capacity constraint
        
    Returns:
        Configured UCB-Constrained algorithm instance
    """
    # Standard discrete price set (arms)
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return UCBConstrainedPricingAlgorithm(
        prices=prices,
        production_capacity=capacity,
        confidence_width=np.sqrt(2),           # Standard UCB1 for rewards
        constraint_confidence_width=np.sqrt(2), # Standard UCB for constraints
        random_seed=42
    )
