# environments/non_stationary.py
"""
environments/non_stationary.py

Highly non-stationary environment for Requirement 3.
Creates adversarial-like conditions where buyer valuations change rapidly over time.

"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class HighlyNonStationaryEnvironment:
    """
    Highly non-stationary environment where buyer valuations change rapidly over time.
    
    This creates an adversarial-like setting to test best-of-both-worlds algorithms.
    The environment switches between different probability distributions frequently,
    making it challenging for algorithms to adapt.
    """
    
    def __init__(self, 
                 prices: List[float],
                 production_capacity: int,
                 total_rounds: int,
                 change_frequency: int = 50,
                 random_seed: Optional[int] = None):
        """
        Initialize highly non-stationary environment
        
        Args:
            prices: List of possible prices
            production_capacity: Maximum production capacity (for interface compatibility)
            total_rounds: Total number of rounds
            change_frequency: How often to change the distribution (rounds)
            random_seed: Random seed for reproducibility
        """
        self.prices = np.array(prices)
        self.production_capacity = production_capacity
        self.total_rounds = total_rounds
        self.change_frequency = change_frequency
        self.current_round = 0
        self.current_distribution_params = {}
        
        # Track distribution history for analysis
        self.distribution_history = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Initialize first distribution
        self._update_distribution()
        
    def _update_distribution(self):
        """
        Update the valuation distribution (happens every change_frequency rounds)
        
        Creates challenging scenarios by switching between:
        - Uniform distributions with different ranges
        - Beta distributions with different shapes  
        - Normal distributions with different parameters
        """
        distribution_type = np.random.choice(['uniform', 'beta', 'normal'])
        
        if distribution_type == 'uniform':
            # Random uniform distributions with varying ranges
            low = np.random.uniform(0.0, 0.4)
            high = np.random.uniform(0.6, 1.0)
            self.current_distribution_params = {
                'type': 'uniform', 
                'low': low, 
                'high': high,
                'optimal_price': (low + high) / 2  # Expected value
            }
            
        elif distribution_type == 'beta':
            # Random beta distributions with different shapes
            alpha = np.random.uniform(0.5, 4.0)
            beta = np.random.uniform(0.5, 4.0)
            expected_value = alpha / (alpha + beta)
            
            self.current_distribution_params = {
                'type': 'beta', 
                'alpha': alpha, 
                'beta': beta,
                'optimal_price': expected_value
            }
            
        else:  # normal
            # Random normal distributions (truncated to [0,1])
            mean = np.random.uniform(0.2, 0.8)
            std = np.random.uniform(0.1, 0.25)
            
            self.current_distribution_params = {
                'type': 'normal', 
                'mean': mean, 
                'std': std,
                'optimal_price': mean  # Approximate (ignoring truncation)
            }
        
        # Record this distribution change
        self.distribution_history.append({
            'round': self.current_round,
            'params': self.current_distribution_params.copy()
        })
            
    def _sample_valuation(self) -> float:
        """
        Sample buyer valuation from current distribution
        
        Returns:
            Buyer valuation in [0, 1]
        """
        params = self.current_distribution_params
        
        if params['type'] == 'uniform':
            return np.random.uniform(params['low'], params['high'])
            
        elif params['type'] == 'beta':
            return np.random.beta(params['alpha'], params['beta'])
            
        else:  # normal
            val = np.random.normal(params['mean'], params['std'])
            return np.clip(val, 0.0, 1.0)  # Truncate to [0,1]
    
    def step(self, selected_prices: Dict[int, float]) -> Tuple[Dict, Dict[int, float], bool]:
        """
        Execute one round of the environment
        
        Args:
            selected_prices: Dictionary {product_id: price}
            
        Returns:
            buyer_info: Information about the buyer and environment state
            rewards: Dictionary {product_id: reward}
            done: Whether the environment is finished
        """
        # Update distribution if needed (creates non-stationarity)
        if self.current_round > 0 and self.current_round % self.change_frequency == 0:
            self._update_distribution()
            
        # Generate buyer valuation from current distribution
        valuation = self._sample_valuation()
        
        # Calculate rewards based on buyer decision
        rewards = {}
        purchases = {}
        
        for product_id, price in selected_prices.items():
            # Buyer purchases if valuation >= price
            purchase = valuation >= price
            reward = price if purchase else 0.0
            
            rewards[product_id] = reward
            purchases[product_id] = purchase
            
        # Create comprehensive buyer info
        buyer_info = {
            "valuations": {0: valuation},  # Single product environment
            "purchases": purchases,
            "round": self.current_round,
            "distribution_params": self.current_distribution_params.copy(),
            "distribution_changed": (self.current_round % self.change_frequency == 0 and self.current_round > 0)
        }
        
        self.current_round += 1
        done = self.current_round >= self.total_rounds
        
        return buyer_info, rewards, done
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_round = 0
        self.distribution_history = []
        self._update_distribution()
        
    def get_current_optimal_price(self) -> float:
        """
        Get optimal price for current distribution (for evaluation purposes)
        
        Returns:
            Theoretical optimal price for current distribution
        """
        return self.current_distribution_params.get('optimal_price', 0.5)
    
    def get_distribution_changes(self) -> List[int]:
        """
        Get list of rounds when distribution changed
        
        Returns:
            List of round numbers when distribution changed
        """
        return [entry['round'] for entry in self.distribution_history]
    
    def get_distribution_history(self) -> List[Dict]:
        """
        Get complete history of distribution changes
        
        Returns:
            List of distribution change events with parameters
        """
        return self.distribution_history.copy()
    
    def get_environment_stats(self) -> Dict:
        """
        Get statistics about the environment
        
        Returns:
            Dictionary with environment statistics
        """
        return {
            "total_rounds": self.total_rounds,
            "change_frequency": self.change_frequency,
            "num_distribution_changes": len(self.distribution_history),
            "current_distribution": self.current_distribution_params.copy(),
            "current_optimal_price": self.get_current_optimal_price()
        }


def create_default_nonstationary_environment(total_rounds: int = 1000,
                                           change_frequency: int = 50,
                                           random_seed: Optional[int] = None) -> HighlyNonStationaryEnvironment:
    """
    Create a default highly non-stationary environment with standard settings
    
    Args:
        total_rounds: Total number of rounds to run
        change_frequency: How often to change distribution
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured HighlyNonStationaryEnvironment
    """
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    return HighlyNonStationaryEnvironment(
        prices=prices,
        production_capacity=100,  # Not directly used, but maintained for interface
        total_rounds=total_rounds,
        change_frequency=change_frequency,
        random_seed=random_seed
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    print("🧪 Testing HighlyNonStationaryEnvironment")
    
    env = create_default_nonstationary_environment(
        total_rounds=200,
        change_frequency=20,
        random_seed=42
    )
    
    print(f"Environment created with {env.change_frequency} round change frequency")
    
    # Simulate a few rounds
    for round_num in range(50):
        # Simulate algorithm selecting a price
        selected_price = 0.5  # Simple fixed price
        
        buyer_info, rewards, done = env.step({0: selected_price})
        
        if buyer_info.get("distribution_changed", False):
            optimal = env.get_current_optimal_price()
            print(f"Round {round_num}: Distribution changed! New optimal ≈ ${optimal:.3f}")
            
        if round_num < 10:  # Show first few rounds
            val = buyer_info["valuations"][0]
            reward = rewards[0]
            print(f"Round {round_num}: valuation=${val:.3f}, reward=${reward:.3f}")
            
    stats = env.get_environment_stats()
    print(f"\nEnvironment stats: {stats}")
    print("✅ Environment test completed successfully!")