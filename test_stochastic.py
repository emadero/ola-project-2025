#!/usr/bin/env python3
"""
Test script for the stochastic environment
Run this to verify that the environment works correctly
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from environments.stochastic import SingleProductStochasticEnvironment, create_default_environment, demo_environment
    print("✅ Successfully imported stochastic environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure environments/__init__.py and environments/stochastic.py exist")
    sys.exit(1)

def test_basic_functionality():
    """Test basic environment functionality"""
    print("\n🧪 Testing Basic Functionality")
    print("-" * 40)
    
    # Create environment
    env = create_default_environment()
    print("✅ Environment created successfully")
    
    # Test reset
    initial_state = env.reset()
    print("✅ Environment reset successfully")
    print(f"   Initial state keys: {list(initial_state.keys())}")
    
    # Test single step
    selected_prices = {0: 0.5}  # Price for product 0
    buyer_info, rewards, done = env.step(selected_prices)
    print("✅ Single step executed successfully")
    print(f"   Buyer valuation: {buyer_info['valuations'][0]:.3f}")
    print(f"   Purchase made: {buyer_info['purchases'][0]}")
    print(f"   Revenue: {rewards[0]:.2f}")
    
    # Test statistics
    stats = env.get_statistics()
    print("✅ Statistics generated successfully")
    
    # Test optimal price
    optimal_price = env.get_optimal_price()
    print(f"✅ Optimal price calculated: {optimal_price}")

def test_different_distributions():
    """Test different valuation distributions"""
    print("\n🎲 Testing Different Distributions")
    print("-" * 40)
    
    distributions = [
        ("uniform", {"low": 0.2, "high": 0.8}),
        ("normal", {"mean": 0.5, "std": 0.15}),
        ("beta", {"alpha": 2.0, "beta": 3.0}),
        ("exponential", {"scale": 0.3})
    ]
    
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for dist_name, params in distributions:
        try:
            env = SingleProductStochasticEnvironment(
                prices=prices,
                production_capacity=5,
                total_rounds=100,
                valuation_distribution=dist_name,
                valuation_params=params,
                random_seed=42
            )
            
            env.reset()
            
            # Run a few steps
            total_revenue = 0
            for _ in range(10):
                price = 0.5  # Fixed price for testing
                buyer_info, rewards, done = env.step({0: price})
                total_revenue += rewards[0]
            
            print(f"✅ {dist_name} distribution works - Total revenue: {total_revenue:.2f}")
            
        except Exception as e:
            print(f"❌ {dist_name} distribution failed: {e}")

def test_oracle_performance():
    """Test oracle (optimal) performance"""
    print("\n🎯 Testing Oracle Performance")
    print("-" * 40)
    
    env = create_default_environment()
    env.reset()
    
    # Get optimal price
    optimal_price = env.get_optimal_price()
    print(f"Optimal price: {optimal_price}")
    
    # Simulate oracle performance
    oracle_rewards = env.simulate_oracle(n_rounds=100)
    oracle_total = sum(oracle_rewards)
    oracle_avg = sum(oracle_rewards) / len(oracle_rewards)
    
    print(f"✅ Oracle simulation completed")
    print(f"   Oracle total revenue (100 rounds): {oracle_total:.2f}")
    print(f"   Oracle average revenue per round: {oracle_avg:.3f}")
    
    # Compare with random pricing
    env.reset()
    random_total = 0
    for _ in range(100):
        random_price = env.prices[env.rng.randint(len(env.prices))]
        buyer_info, rewards, done = env.step({0: random_price})
        random_total += rewards[0]
    
    random_avg = random_total / 100
    print(f"   Random pricing average revenue: {random_avg:.3f}")
    print(f"   Oracle advantage: {(oracle_avg - random_avg):.3f} ({((oracle_avg/random_avg - 1)*100):.1f}%)")

def main():
    """Run all tests"""
    print("🎮 Testing Single Product Stochastic Environment")
    print("=" * 60)
    
    test_basic_functionality()
    test_different_distributions()
    test_oracle_performance()
    
    print("\n🎉 All tests completed!")
    print("\n💡 To see a visual demo, run:")
    print("   python -c \"from environments.stochastic import demo_environment; demo_environment()\"")

if __name__ == "__main__":
    main()