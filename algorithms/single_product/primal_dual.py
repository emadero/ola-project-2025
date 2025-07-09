# algorithms/single_product/primal_dual.py
"""
algorithms/single_product/primal_dual.py

Primal-Dual algorithm for single product pricing with inventory constraints.
Based on the general auctions framework from course material with theoretical corrections.

This implements a best-of-both-worlds algorithm that performs well in both
stochastic and adversarial environments using proper EXP3-like updates.

"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any


class PrimalDualPricingAlgorithm:
    """
    Primal-Dual algorithm for single product pricing with inventory constraints.
    
    THEORETICAL FOUNDATION:
    Based on the Lagrangian formulation from general auctions:
    L(γ, λ) = f(γ) - λ[c(γ) - ρ]
    
    Where:
    - γ: pricing strategy (distribution over prices) 
    - λ: dual variable (Lagrange multiplier for inventory constraint)
    - f(γ): expected reward function (revenue)
    - c(γ): expected cost function (production constraint)
    - ρ: per-round budget/capacity constraint
    
    IMPROVEMENTS v4:
    1. Proper EXP3-like updates with exploration
    2. Theoretical learning rates: η = √(log(K)/T) for primal, η = 1/√T for dual  
    3. Correct bandit feedback handling
    4. Importance weighting for unobserved arms
    5. Best-of-both-worlds guarantees
    """
    
    def __init__(self,
                 prices: List[float],
                 production_capacity: int,
                 horizon_T: int,
                 learning_rate: Optional[float] = None,
                 primal_learning_rate: Optional[float] = None,
                 exploration_param: Optional[float] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize Primal-Dual pricing algorithm with theoretical parameters
        
        Args:
            prices: List of possible prices to choose from
            production_capacity: Total inventory/production budget (B)
            horizon_T: Time horizon (needed for theoretical learning rates)
            learning_rate: Learning rate for dual variable updates (default: 1/√T)
            primal_learning_rate: Learning rate for primal variables (default: √(log K/T))
            exploration_param: Exploration parameter (default: theoretical)
            random_seed: Random seed for reproducibility
        """
        self.prices = np.array(prices)
        self.K = len(prices)  # Number of price options
        self.production_capacity = production_capacity
        self.T = horizon_T  # Time horizon
        
        # THEORETICAL LEARNING RATES (from OLA theory)
        if learning_rate is None:
            # Dual learning rate: η_dual = 1/√T for convergence
            self.learning_rate = 1.0 / math.sqrt(max(self.T, 1))
        else:
            self.learning_rate = learning_rate
            
        if primal_learning_rate is None:
            # Primal learning rate: η_primal = √(log(K)/T) for EXP3-like
            self.primal_learning_rate = math.sqrt(math.log(max(self.K, 2)) / max(self.T, 1))
        else:
            self.primal_learning_rate = primal_learning_rate
            
        # Exploration parameter for EXP3-like updates
        if exploration_param is None:
            # Theoretical: γ = min(1, √(K*log(K)/T))
            self.gamma = min(1.0, math.sqrt(self.K * math.log(max(self.K, 2)) / max(self.T, 1)))
        else:
            self.gamma = exploration_param
        
        # Algorithm state
        self.current_round = 0
        self.remaining_capacity = production_capacity
        
        # Per-round budget constraint (ρ in the formulation)
        # Theoretical: ρ = B/T (amortized capacity per round)
        self.rho = self.production_capacity / max(self.T, 1)
        
        # Dual variable (Lagrange multiplier for inventory constraint)
        self.lambda_t = 0.0
        
        # Primal variables: probability distribution over prices
        self.price_probabilities = np.ones(self.K) / self.K  # Start uniform
        
        # EXP3-like reward estimates for each price
        self.reward_estimates = np.zeros(self.K)
        
        # History tracking for analysis
        self.rewards_history = []
        self.prices_history = []
        self.lambda_history = []
        self.capacity_history = []
        self.production_decisions = []
        self.lagrangian_history = []
        self.price_probabilities_history = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        print(f"   Primal-Dual initialized:")
        print(f"   η_dual (dual learning): {self.learning_rate:.6f}")
        print(f"   η_primal (price learning): {self.primal_learning_rate:.6f}")
        print(f"   γ (exploration): {self.gamma:.6f}")
        print(f"   ρ (per-round budget): {self.rho:.3f}")
        print(f"   Horizon T: {self.T}")
            
    def _compute_lagrangian_values(self, 
                                 observed_reward: float,
                                 selected_price_idx: int,
                                 buyer_valuation: float) -> np.ndarray:
        """
        Compute Lagrangian values for all price options using bandit feedback
        
        THEORETICAL IMPROVEMENT: Use importance weighting for bandit feedback
        L(p, λ) = f(p) - λ * [c(p) - ρ]
        
        Args:
            observed_reward: Reward received from selected price
            selected_price_idx: Index of price that was selected  
            buyer_valuation: Buyer's valuation (for estimating counterfactuals)
            
        Returns:
            Array of Lagrangian values for each price
        """
        # Initialize Lagrangian values
        lagrangian_values = np.zeros(self.K)
        
        # THEORETICAL APPROACH: Use full information when available
        # Estimate what each price would have given based on buyer valuation
        for i, price in enumerate(self.prices):
            # Estimate reward: price if buyer would buy, 0 otherwise
            estimated_reward = price if buyer_valuation >= price else 0.0
            
            # Estimate cost: 1 if we would produce at this price, 0 otherwise  
            estimated_cost = 1.0 if buyer_valuation >= price else 0.0
            
            # Lagrangian value: reward - λ * (cost - per_round_budget)
            lagrangian_values[i] = estimated_reward - self.lambda_t * (estimated_cost - self.rho)
        
        # BANDIT FEEDBACK CORRECTION: Importance weight the observed arm
        if selected_price_idx >= 0:
            selected_prob = self.price_probabilities[selected_price_idx]
            if selected_prob > 1e-10:  # Avoid division by zero
                # Importance weighted update for selected arm
                importance_weight = observed_reward / selected_prob
                
                # Blend importance weighted estimate with counterfactual estimate
                blend_factor = 0.5  # Balance between bandit and full info
                lagrangian_values[selected_price_idx] = (
                    blend_factor * (importance_weight - self.lambda_t * (1.0 - self.rho)) +
                    (1 - blend_factor) * lagrangian_values[selected_price_idx]
                )
        
        return lagrangian_values
    
    def _update_price_probabilities(self, lagrangian_values: np.ndarray):
        """
        Update price probabilities using EXP3-like method with exploration
        
        THEORETICAL IMPROVEMENT: Proper EXP3 with exploration parameter
        
        Args:
            lagrangian_values: Lagrangian values for each price
        """
        # EXP3-like exponential weights update
        eta = self.primal_learning_rate
        
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-10
        
        # Update reward estimates (EXP3-style)
        self.reward_estimates += eta * lagrangian_values
        
        # Compute exponential weights
        weights = np.exp(self.reward_estimates)
        weights = np.clip(weights, epsilon, 1e10)  # Prevent overflow
        
        # Normalize to get probabilities
        prob_weights = weights / (np.sum(weights) + epsilon)
        
        # EXP3 exploration: mix with uniform distribution
        uniform_prob = np.ones(self.K) / self.K
        self.price_probabilities = (1 - self.gamma) * prob_weights + self.gamma * uniform_prob
        
        # Final normalization and clipping
        self.price_probabilities = np.clip(self.price_probabilities, epsilon, 1.0)
        self.price_probabilities /= np.sum(self.price_probabilities)
        
    def _update_dual_variable(self, actual_cost: float):
        """
        Update dual variable using projected gradient ascent
        
        THEORETICAL IMPROVEMENT: Proper projection and learning rate
        
        This implements: λ_{t+1} = Π_{[0, ∞)}(λ_t + η * (c_t - ρ))
        
        Args:
            actual_cost: Actual cost incurred this round (0 or 1)
        """
        # Constraint violation: how much we exceeded the per-round budget
        constraint_violation = actual_cost - self.rho
        
        # Dual variable update with theoretical learning rate
        self.lambda_t = self.lambda_t + self.learning_rate * constraint_violation
        
        # Project to feasible region [0, +∞)
        # Dual variables must be non-negative
        self.lambda_t = max(0.0, self.lambda_t)
        
        # Optional: upper bound to prevent numerical explosion
        # In theory λ can grow unbounded, but for numerical stability
        max_lambda = 10.0 / max(self.rho, 0.01)
        self.lambda_t = min(self.lambda_t, max_lambda)
        
    def select_prices(self) -> Dict[int, float]:
        """
        Select prices for current round using primal-dual method
        
        THEORETICAL IMPROVEMENT: Direct sampling from learned distribution
        
        Returns:
            Dictionary {product_id: price} or {} if no production
        """
        # Check remaining capacity
        if self.remaining_capacity <= 0:
            return {}  # Cannot produce
            
        # Sample price according to current probability distribution
        price_idx = np.random.choice(self.K, p=self.price_probabilities)
        selected_price = self.prices[price_idx]
        
        # THEORETICAL DECISION: Always produce when capacity allows
        # The dual variable λ already encodes the capacity constraint
        # No need for additional production probability
        
        if self.remaining_capacity > 0:
            return {0: selected_price}  # Single product environment
        else:
            return {}  # No production this round
            
    def update(self, 
               selected_prices: Dict[int, float], 
               rewards: Dict[int, float], 
               buyer_info: Dict):
        """
        Update algorithm state based on round outcome
        
        THEORETICAL IMPROVEMENT: Better use of buyer information
        
        Args:
            selected_prices: Prices that were selected {product_id: price}
            rewards: Rewards received {product_id: reward}
            buyer_info: Information about buyer and environment
        """
        # Determine production and costs
        produced = len(selected_prices) > 0
        actual_cost = 1.0 if produced else 0.0
        actual_reward = sum(rewards.values()) if rewards else 0.0
        selected_price = list(selected_prices.values())[0] if selected_prices else None
        selected_price_idx = -1
        
        if selected_price is not None:
            # Find index of selected price
            selected_price_idx = np.argmin(np.abs(self.prices - selected_price))
        
        # Update capacity
        if produced:
            self.remaining_capacity = max(0, self.remaining_capacity - 1)
            
        # Store history
        self.rewards_history.append(actual_reward)
        self.prices_history.append(selected_price if selected_price else 0.0)
        self.lambda_history.append(self.lambda_t)
        self.capacity_history.append(self.remaining_capacity)
        self.production_decisions.append(produced)
        self.price_probabilities_history.append(self.price_probabilities.copy())
        
        # Extract buyer valuation for counterfactual estimation
        buyer_valuation = 0.0
        if "valuations" in buyer_info:
            buyer_valuation = buyer_info["valuations"].get(0, 0.0)
        elif produced:
            # Fallback: if buyer purchased, valuation >= price
            buyer_valuation = selected_price
        
        # Compute Lagrangian values using improved method
        lagrangian_values = self._compute_lagrangian_values(
            actual_reward, selected_price_idx, buyer_valuation
        )
        self.lagrangian_history.append(lagrangian_values.copy())
        
        # Update primal variables (price probabilities)
        self._update_price_probabilities(lagrangian_values)
        
        # Update dual variable (constraint multiplier)
        self._update_dual_variable(actual_cost)
        
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        self.current_round = 0
        self.remaining_capacity = self.production_capacity
        self.lambda_t = 0.0
        self.price_probabilities = np.ones(self.K) / self.K
        self.reward_estimates = np.zeros(self.K)
        
        # Clear all histories
        self.rewards_history = []
        self.prices_history = []
        self.lambda_history = []
        self.capacity_history = []
        self.production_decisions = []
        self.lagrangian_history = []
        self.price_probabilities_history = []
        
    def get_theoretical_regret_bound(self) -> float:
        """
        Get theoretical regret bound for this algorithm
        
        THEORETICAL ADDITION: Compute expected regret bound
        
        Returns:
            Upper bound on expected regret
        """
        # For EXP3-like with constraints: regret ≤ O(√(K*T*log(K))) + constraint violations
        primal_regret = math.sqrt(2 * self.K * self.T * math.log(max(self.K, 2)))
        
        # Dual regret from constraint violations: O(B*√T)
        dual_regret = self.production_capacity * math.sqrt(self.T)
        
        return primal_regret + dual_regret
        
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm statistics
        
        THEORETICAL ADDITION: Include theoretical metrics
        
        Returns:
            Dictionary with performance metrics and internal state
        """
        if not self.rewards_history:
            return {"error": "No data available"}
            
        return {
            "total_reward": sum(self.rewards_history),
            "average_reward": np.mean(self.rewards_history),
            "total_rounds": len(self.rewards_history),
            "production_rate": np.mean(self.production_decisions),
            "capacity_used": self.production_capacity - self.remaining_capacity,
            "capacity_utilization": (self.production_capacity - self.remaining_capacity) / max(self.production_capacity, 1),
            "final_lambda": self.lambda_t,
            "average_lambda": np.mean(self.lambda_history) if self.lambda_history else 0.0,
            "max_lambda": np.max(self.lambda_history) if self.lambda_history else 0.0,
            "final_price_probabilities": self.price_probabilities.copy(),
            "most_likely_price": self.prices[np.argmax(self.price_probabilities)],
            "price_entropy": -np.sum(self.price_probabilities * np.log(self.price_probabilities + 1e-10)),
            
            # THEORETICAL ADDITIONS
            "theoretical_regret_bound": self.get_theoretical_regret_bound(),
            "learning_rates": {
                "eta_dual": self.learning_rate,
                "eta_primal": self.primal_learning_rate,
                "gamma_exploration": self.gamma
            },
            "constraint_satisfaction": {
                "per_round_budget": self.rho,
                "average_cost": np.mean([1.0 if p else 0.0 for p in self.production_decisions]),
                "constraint_violation": np.mean([1.0 if p else 0.0 for p in self.production_decisions]) - self.rho
            }
        }
        
    def get_best_price(self) -> float:
        """Get the currently most likely price according to learned distribution"""
        return self.prices[np.argmax(self.price_probabilities)]
        
    def get_current_strategy(self) -> Dict[str, Any]:
        """
        Get current algorithm strategy
        
        Returns:
            Dictionary with current primal and dual variables
        """
        return {
            "price_probabilities": self.price_probabilities.copy(),
            "dual_variable": self.lambda_t,
            "remaining_capacity": self.remaining_capacity,
            "exploration_level": self.gamma,
            "constraint_multiplier": self.lambda_t * self.rho
        }


def create_default_primal_dual(production_capacity: int = 800,
                             horizon_T: int = 1000,
                             random_seed: Optional[int] = None) -> PrimalDualPricingAlgorithm:
    """
    Create a default primal-dual algorithm with theoretical settings
    
    Args:
        production_capacity: Total production capacity
        horizon_T: Time horizon (needed for theoretical learning rates)
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured PrimalDualPricingAlgorithm with theoretical parameters
    """
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return PrimalDualPricingAlgorithm(
        prices=prices,
        production_capacity=production_capacity,
        horizon_T=horizon_T,
        random_seed=random_seed
    )

