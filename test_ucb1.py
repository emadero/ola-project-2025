#!/usr/bin/env python3
"""
Test script for corrected UCB1 algorithm (Multi-Armed Bandit approach)
Run this to verify that UCB1 works correctly with the stochastic environment
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from algorithms.single_product.ucb import UCB1PricingAlgorithm, create_default_ucb1, demo_ucb1
    from environments.stochastic import create_default_environment
    print("✅ Successfully imported corrected UCB1 algorithm and stochastic environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all required files exist:")
    print("   - algorithms/__init__.py")
    print("   - algorithms/single_product/ucb.py") 
    print("   - environments/__init__.py")
    print("   - environments/stochastic.py")
    sys.exit(1)

def test_ucb1_multi_armed_bandit_structure():
    """Test that UCB1 follows Multi-Armed Bandit structure"""
    print("\n🧪 Testing UCB1 Multi-Armed Bandit Structure")
    print("-" * 50)
    
    # Create UCB1 algorithm
    ucb1 = create_default_ucb1()
    print("✅ UCB1 algorithm created successfully")
    
    # Test Multi-Armed Bandit structure
    assert hasattr(ucb1, 'arm_counts'), "Missing arm_counts"
    assert hasattr(ucb1, 'arm_means'), "Missing arm_means"
    assert hasattr(ucb1, 'arm_rewards'), "Missing arm_rewards"
    print("✅ Has Multi-Armed Bandit structure")
    
    # Test arm-price mapping
    assert len(ucb1.arm_counts) == len(ucb1.prices), "Arm count should match number of prices"
    assert len(ucb1.arm_means) == len(ucb1.prices), "Arm means should match number of prices"
    print("✅ Each price is treated as separate arm")
    
    # Test arm ID conversion
    for i, price in enumerate(ucb1.prices):
        arm_id = ucb1._price_to_arm_id(price)
        converted_price = ucb1._arm_id_to_price(arm_id)
        assert arm_id == i, f"Arm ID conversion failed for price {price}"
        assert abs(converted_price - price) < 1e-6, f"Price conversion failed for arm {arm_id}"
    print("✅ Arm-price conversion works correctly")

def test_ucb1_basic_functionality():
    """Test basic UCB1 functionality"""
    print("\n🧪 Testing UCB1 Basic Functionality")
    print("-" * 50)
    
    # Create UCB1 algorithm
    ucb1 = create_default_ucb1()
    
    # Test price selection
    selected_prices = ucb1.select_prices()
    print(f"✅ Price selection works: {selected_prices}")
    assert len(selected_prices) == 1, "Should select exactly one price"
    assert 0 in selected_prices, "Should select for product 0"
    
    # Test update with Multi-Armed Bandit logic
    buyer_info = {
        "valuations": {0: 0.7},
        "purchases": {0: True},
        "round": 0
    }
    rewards = {0: selected_prices[0]}
    
    # Check initial arm statistics
    selected_price = selected_prices[0]
    arm_id = ucb1._price_to_arm_id(selected_price)
    initial_count = ucb1.arm_counts[arm_id]
    
    ucb1.update(selected_prices, rewards, buyer_info)
    
    # Verify arm statistics were updated
    assert ucb1.arm_counts[arm_id] == initial_count + 1, "Arm count should increment"
    assert ucb1.arm_means[arm_id] == selected_price, "Arm mean should equal reward"
    print("✅ Algorithm update works with Multi-Armed Bandit logic")
    
    # Test arm statistics
    arm_stats = ucb1.get_arm_statistics()
    print(f"✅ Arm statistics generated: {len(arm_stats)} arms")
    assert len(arm_stats) == len(ucb1.prices), "Should have statistics for all arms"
    
    # Test UCB scores
    scores = ucb1.get_ucb_scores()
    print(f"✅ UCB scores calculated for {len(scores)} prices")
    assert len(scores) == len(ucb1.prices), "Should have UCB scores for all prices"

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
        # UCB1 selects price (arm)
        selected_prices = ucb1.select_prices()
        
        # Environment responds
        buyer_info, rewards, done = env.step(selected_prices)
        
        # UCB1 learns from feedback
        ucb1.update(selected_prices, rewards, buyer_info)
        
        round_revenue = rewards[0]
        total_revenue += round_revenue
        
        if round_num < 5:  # Show first 5 rounds
            selected_arm = ucb1._price_to_arm_id(selected_prices[0])
            print(f"  Round {round_num + 1}: arm={selected_arm}, price={selected_prices[0]:.2f}, "
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
    
    # Show UCB1 learning with Multi-Armed Bandit perspective
    print(f"\n🧠 UCB1 Multi-Armed Bandit Learning Analysis:")
    best_arm = ucb1.get_best_arm()
    best_price = ucb1.get_best_price()
    print(f"   UCB1 best arm: {best_arm} (price ${best_price:.2f})")
    
    stats = ucb1.get_algorithm_stats()
    print(f"   Exploration ratio: {stats['exploration_ratio']:.3f}")
    print(f"   Most selected arm: {stats['most_selected_arm']} "
          f"(price ${stats['most_selected_price']:.2f}, selected {stats['most_selected_count']} times)")
    
    # Show arm statistics for top 3 arms
    arm_stats = ucb1.get_arm_statistics()
    sorted_arms = sorted(arm_stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True)[:3]
    print(f"   Top 3 arms by mean reward:")
    for arm_name, stats in sorted_arms:
        if stats['times_selected'] > 0:
            print(f"     {arm_name}: mean_reward={stats['mean_reward']:.3f}, "
                  f"times_selected={stats['times_selected']}")

def test_ucb1_convergence():
    """Test UCB1 convergence to optimal arm"""
    print("\n📈 Testing UCB1 Convergence")
    print("-" * 50)
    
    env = create_default_environment()
    ucb1 = create_default_ucb1()
    
    env.reset()
    
    # Run longer experiment
    n_rounds = 500
    revenues = []
    selected_arms_history = []
    
    print(f"Running {n_rounds} rounds to test convergence...")
    
    for round_num in range(n_rounds):
        selected_prices = ucb1.select_prices()
        buyer_info, rewards, done = env.step(selected_prices)
        ucb1.update(selected_prices, rewards, buyer_info)
        
        revenues.append(rewards[0])
        selected_arm = ucb1._price_to_arm_id(selected_prices[0])
        selected_arms_history.append(selected_arm)
    
    # Analyze convergence
    optimal_price = env.get_optimal_price()
    optimal_arm = ucb1._price_to_arm_id(optimal_price)
    
    # Look at last 100 rounds
    recent_arms = selected_arms_history[-100:]
    recent_revenues = revenues[-100:]
    
    # Calculate how often UCB1 selected optimal or near-optimal arms
    optimal_arm_count = sum(1 for arm in recent_arms if arm == optimal_arm)
    optimal_arm_ratio = optimal_arm_count / len(recent_arms)
    
    # Near-optimal arms (within 1 arm of optimal)
    near_optimal_count = sum(1 for arm in recent_arms 
                           if abs(arm - optimal_arm) <= 1)
    near_optimal_ratio = near_optimal_count / len(recent_arms)
    
    print(f"📊 Convergence Analysis (last 100 rounds):")
    print(f"   Optimal arm: {optimal_arm} (price ${optimal_price:.2f})")
    print(f"   Optimal arm selections: {optimal_arm_ratio:.1%}")
    print(f"   Near-optimal arm selections: {near_optimal_ratio:.1%}")
    print(f"   Average revenue: {np.mean(recent_revenues):.3f}")
    
    # Show exploration decay
    stats = ucb1.get_algorithm_stats()
    print(f"\n🔍 Final Exploration Stats:")
    print(f"   Total exploration: {stats['exploration_count']}")
    print(f"   Total exploitation: {stats['exploitation_count']}")
    print(f"   Final exploration ratio: {stats['exploration_ratio']:.3f}")
    
    # Show final arm rankings
    arm_stats = ucb1.get_arm_statistics()
    sorted_arms = sorted(arm_stats.items(), key=lambda x: x[1]['mean_reward'], reverse=True)[:5]
    print(f"\n🏆 Top 5 Arms by Mean Reward:")
    for rank, (arm_name, stats) in enumerate(sorted_arms, 1):
        if stats['times_selected'] > 0:
            print(f"   {rank}. {arm_name}: mean_reward={stats['mean_reward']:.3f}, "
                  f"times_selected={stats['times_selected']}")

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
        
        ucb1 = UCB1PricingAlgorithm(
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

def test_regret_bounds():
    """Test theoretical regret bounds calculation"""
    print("\n📐 Testing Theoretical Regret Bounds")
    print("-" * 50)
    
    ucb1 = create_default_ucb1()
    
    # Test initial regret bounds
    bounds = ucb1.get_regret_bounds()
    assert "theoretical_regret_bound" in bounds, "Missing regret bound"
    assert "number_of_arms" in bounds, "Missing number of arms"
    print("✅ Initial regret bounds calculated")
    
    # Run some rounds and test bounds
    env = create_default_environment()
    env.reset()
    
    for _ in range(50):
        selected_prices = ucb1.select_prices()
        buyer_info, rewards, done = env.step(selected_prices)
        ucb1.update(selected_prices, rewards, buyer_info)
    
    bounds = ucb1.get_regret_bounds()
    assert bounds["theoretical_regret_bound"] > 0, "Should have positive regret bound after rounds"
    assert bounds["number_of_arms"] == len(ucb1.prices), "Number of arms should match prices"
    
    print(f"✅ Regret bounds after 50 rounds:")
    print(f"   Theoretical bound: {bounds['theoretical_regret_bound']:.2f}")
    print(f"   Per-round bound: {bounds['regret_bound_per_round']:.3f}")
    print(f"   Number of arms: {bounds['number_of_arms']}")

def main():
    """Run all UCB1 tests"""
    print("🎯 Testing Corrected UCB1 Algorithm (Multi-Armed Bandit)")
    print("=" * 70)
    
    test_ucb1_multi_armed_bandit_structure()
    test_ucb1_basic_functionality()
    test_ucb1_with_environment()
    test_ucb1_convergence() 
    test_different_confidence_widths()
    test_regret_bounds()
    
    print("\n🎉 All UCB1 tests completed!")
    print("\n💡 To see a visual demo, run:")
    print("   python -c \"from algorithms.single_product.ucb import demo_ucb1; demo_ucb1()\"")

if __name__ == "__main__":
    main()