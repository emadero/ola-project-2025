#!/usr/bin/env python3
"""
Test script for UCB1 algorithm
Run this to verify that UCB1 works correctly with the stochastic environment
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from algorithms.single_product.ucb import UCB1Algorithm, create_default_ucb1, demo_ucb1
    from environments.stochastic import create_default_environment
    print("✅ Successfully imported UCB1 algorithm and stochastic environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all required files exist:")
    print("   - algorithms/__init__.py")
    print("   - algorithms/single_product/ucb.py") 
    print("   - environments/__init__.py")
    print("   - environments/stochastic.py")
    sys.exit(1)

def test_ucb1_basic_functionality():
    """Test basic UCB1 functionality"""
    print("\n🧪 Testing UCB1 Basic Functionality")
    print("-" * 50)
    
    # Create UCB1 algorithm
    ucb1 = create_default_ucb1()
    print("✅ UCB1 algorithm created successfully")
    
    # Test price selection
    selected_prices = ucb1.select_prices()
    print(f"✅ Price selection works: {selected_prices}")
    
    # Test update
    buyer_info = {
        "valuations": {0: 0.7},
        "purchases": {0: True},
        "round": 0
    }
    rewards = {0: selected_prices[0]}
    
    ucb1.update(selected_prices, rewards, buyer_info)
    print("✅ Algorithm update works")
    
    # Test statistics
    stats = ucb1.get_algorithm_stats()
    print(f"✅ Statistics generated: {len(stats)} metrics")
    
    # Test UCB scores
    scores = ucb1.get_ucb_scores()
    print(f"✅ UCB scores calculated for {len(scores)} prices")

def test_ucb1_with_environment():
    """Test UCB1 integrated with stochastic environment"""
    print("\n🎮 Testing UCB1 with Stochastic Environment")
    print("-" * 50)
    
    # Create environment and algorithm
    env = create_default_environment()
    ucb1 = create_default_ucb1()
    
    # Reset environment
    env.reset()
    
    print("Running 20 rounds of UCB1 vs Environment:")
    
    total_revenue = 0
    for round_num in range(20):
        # UCB1 selects price
        selected_prices = ucb1.select_prices()
        
        # Environment responds
        buyer_info, rewards, done = env.step(selected_prices)
        
        # UCB1 learns from feedback
        ucb1.update(selected_prices, rewards, buyer_info)
        
        round_revenue = rewards[0]
        total_revenue += round_revenue
        
        if round_num < 5:  # Show first 5 rounds
            print(f"  Round {round_num + 1}: price={selected_prices[0]:.2f}, "
                  f"valuation={buyer_info['valuations'][0]:.3f}, "
                  f"bought={buyer_info['purchases'][0]}, "
                  f"revenue={round_revenue:.2f}")
    
    average_revenue = total_revenue / 20
    print(f"\n📊 Results after 20 rounds:")
    print(f"   Total revenue: {total_revenue:.2f}")
    print(f"   Average revenue per round: {average_revenue:.3f}")
    
    # Compare with optimal strategy
    optimal_price = env.get_optimal_price()
    oracle_rewards = env.simulate_oracle(n_rounds=20)
    oracle_average = np.mean(oracle_rewards)
    
    print(f"\n🎯 Comparison with Oracle:")
    print(f"   Oracle optimal price: {optimal_price:.2f}")
    print(f"   Oracle average revenue: {oracle_average:.3f}")
    print(f"   UCB1 performance: {(average_revenue/oracle_average)*100:.1f}% of optimal")
    
    # Show UCB1 learning
    print(f"\n🧠 UCB1 Learning Analysis:")
    best_price = ucb1.get_best_price()
    print(f"   UCB1 best discovered price: {best_price:.2f}")
    
    stats = ucb1.get_algorithm_stats()
    print(f"   Exploration ratio: {stats['exploration_ratio']:.3f}")
    print(f"   Most selected price: {stats['most_selected_price']:.2f} "
          f"(selected {stats['most_selected_count']} times)")

def test_ucb1_convergence():
    """Test UCB1 convergence to optimal price"""
    print("\n📈 Testing UCB1 Convergence")
    print("-" * 50)
    
    env = create_default_environment()
    ucb1 = create_default_ucb1()
    
    env.reset()
    
    # Run longer experiment
    n_rounds = 500
    revenues = []
    selected_prices_history = []
    
    print(f"Running {n_rounds} rounds to test convergence...")
    
    for round_num in range(n_rounds):
        selected_prices = ucb1.select_prices()
        buyer_info, rewards, done = env.step(selected_prices)
        ucb1.update(selected_prices, rewards, buyer_info)
        
        revenues.append(rewards[0])
        selected_prices_history.append(selected_prices[0])
    
    # Analyze convergence
    optimal_price = env.get_optimal_price()
    
    # Look at last 100 rounds
    recent_prices = selected_prices_history[-100:]
    recent_revenues = revenues[-100:]
    
    # Calculate how often UCB1 selected near-optimal prices
    tolerance = 0.1  # Within 0.1 of optimal price
    near_optimal_count = sum(1 for p in recent_prices 
                           if abs(p - optimal_price) <= tolerance)
    near_optimal_ratio = near_optimal_count / len(recent_prices)
    
    print(f"📊 Convergence Analysis (last 100 rounds):")
    print(f"   Optimal price: {optimal_price:.2f}")
    print(f"   Average selected price: {np.mean(recent_prices):.3f}")
    print(f"   Price standard deviation: {np.std(recent_prices):.3f}")
    print(f"   Near-optimal selections: {near_optimal_ratio:.1%}")
    print(f"   Average revenue: {np.mean(recent_revenues):.3f}")
    
    # Show exploration decay
    stats = ucb1.get_algorithm_stats()
    print(f"\n🔍 Final Exploration Stats:")
    print(f"   Total exploration: {stats['exploration_count']}")
    print(f"   Total exploitation: {stats['exploitation_count']}")
    print(f"   Final exploration ratio: {stats['exploration_ratio']:.3f}")

def test_different_confidence_widths():
    """Test UCB1 with different confidence width parameters"""
    print("\n⚙️ Testing Different Confidence Widths")
    print("-" * 50)
    
    prices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    confidence_widths = [0.5, 1.0, np.sqrt(2), 2.0, 3.0]
    
    env = create_default_environment()
    n_rounds = 100
    
    print(f"Testing {len(confidence_widths)} different confidence widths:")
    
    for width in confidence_widths:
        env.reset()
        
        ucb1 = UCB1Algorithm(
            prices=prices,
            production_capacity=10,
            confidence_width=width,
            random_seed=42
        )
        
        total_revenue = 0
        for _ in range(n_rounds):
            selected_prices = ucb1.select_prices()
            buyer_info, rewards, done = env.step(selected_prices)
            ucb1.update(selected_prices, rewards, buyer_info)
            total_revenue += rewards[0]
        
        avg_revenue = total_revenue / n_rounds
        stats = ucb1.get_algorithm_stats()
        
        print(f"   Width {width:.3f}: avg_revenue={avg_revenue:.3f}, "
              f"exploration_ratio={stats['exploration_ratio']:.3f}")

def main():
    """Run all UCB1 tests"""
    print("🎯 Testing UCB1 Algorithm for Single Product Pricing")
    print("=" * 70)
    
    test_ucb1_basic_functionality()
    test_ucb1_with_environment()
    test_ucb1_convergence() 
    test_different_confidence_widths()
    
    print("\n🎉 All UCB1 tests completed!")
    print("\n💡 To see a visual demo, run:")
    print("   python -c \"from algorithms.single_product.ucb import demo_ucb1; demo_ucb1()\"")

if __name__ == "__main__":
    main()