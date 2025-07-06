#!/usr/bin/env python3
"""
Test script for corrected UCB-Constrained algorithm (Auction-like approach)
Run this to verify that UCB-Constrained follows the auction theory paradigm
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from algorithms.single_product.constrained_ucb import UCBConstrainedPricingAlgorithm, create_default_constrained_ucb1, demo_constrained_ucb1
    from algorithms.single_product.ucb import UCB1PricingAlgorithm, create_default_ucb1
    from environments.stochastic import create_default_environment
    print("✅ Successfully imported corrected constrained algorithm")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all required files exist:")
    print("   - algorithms/single_product/constrained_ucb.py")
    print("   - algorithms/single_product/ucb.py")
    print("   - environments/stochastic.py")
    sys.exit(1)

def test_auction_theory_structure():
    """Test that UCB-Constrained follows auction theory structure"""
    print("\n🧪 Test 1: Auction Theory Structure")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=5)
    print("✅ UCB-Constrained algorithm created successfully")
    
    # Test inheritance from UCB1 (Multi-Armed Bandit)
    assert isinstance(constrained_ucb, UCB1PricingAlgorithm), "Should inherit from UCB1"
    assert hasattr(constrained_ucb, 'arm_counts'), "Should have UCB1 arm structure"
    assert hasattr(constrained_ucb, 'arm_means'), "Should have UCB1 arm means"
    print("✅ Correctly inherits Multi-Armed Bandit structure")
    
    # Test constraint-specific structure
    assert hasattr(constrained_ucb, 'constraint_counts'), "Missing constraint statistics"
    assert hasattr(constrained_ucb, 'constraint_means'), "Missing constraint means"
    assert hasattr(constrained_ucb, 'constraint_totals'), "Missing constraint totals"
    print("✅ Has constraint statistics structure")
    
    # Test dual confidence bounds methods
    assert hasattr(constrained_ucb, '_calculate_reward_ucb_bounds'), "Missing reward UCB bounds"
    assert hasattr(constrained_ucb, '_calculate_constraint_lcb_bounds'), "Missing constraint LCB bounds"
    assert hasattr(constrained_ucb, '_find_feasible_arms'), "Missing feasibility check"
    print("✅ Has auction theory dual bounds methods")
    
    # Test capacity tracking
    assert constrained_ucb.remaining_capacity == 5, "Wrong initial capacity"
    assert constrained_ucb.total_capacity == 5, "Wrong total capacity"
    print("✅ Capacity tracking initialized correctly")

def test_dual_confidence_bounds():
    """Test the dual confidence bounds calculation"""
    print("\n🧪 Test 2: Dual Confidence Bounds")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=10)
    
    # Test initial bounds calculation
    reward_bounds = constrained_ucb._calculate_reward_ucb_bounds()
    constraint_bounds = constrained_ucb._calculate_constraint_lcb_bounds()
    
    assert len(reward_bounds) == len(constrained_ucb.prices), "Reward bounds for all arms"
    assert len(constraint_bounds) == len(constrained_ucb.prices), "Constraint bounds for all arms"
    print("✅ Dual confidence bounds calculated for all arms")
    
    # Test bounds properties
    for arm_id in range(len(constrained_ucb.prices)):
        assert reward_bounds[arm_id] >= 0, "Reward bounds should be non-negative"
        assert constraint_bounds[arm_id] >= 0, "Constraint bounds should be non-negative"
    print("✅ Bounds have correct properties")
    
    # Give algorithm some data and test bound updates
    test_scenarios = [
        (0.3, 0.5, True),   # price, valuation, purchase
        (0.4, 0.6, True),
        (0.2, 0.1, False),
    ]
    
    for round_num, (price, valuation, purchase) in enumerate(test_scenarios):
        selected_prices = {0: price}
        reward = price if purchase else 0.0
        buyer_info = {"round": round_num}
        constrained_ucb.update(selected_prices, {0: reward}, buyer_info)
    
    # Test bounds after learning
    updated_reward_bounds = constrained_ucb._calculate_reward_ucb_bounds()
    updated_constraint_bounds = constrained_ucb._calculate_constraint_lcb_bounds()
    
    # Check that used arms have finite bounds
    for round_num, (price, _, _) in enumerate(test_scenarios):
        arm_id = constrained_ucb._price_to_arm_id(price)
        assert updated_reward_bounds[arm_id] != float('inf'), f"Used arm {arm_id} should have finite reward bound"
        assert updated_constraint_bounds[arm_id] < float('inf'), f"Used arm {arm_id} should have finite constraint bound"
    
    print("✅ Bounds update correctly after learning")

def test_feasibility_and_optimization():
    """Test feasibility checking and constrained optimization"""
    print("\n🧪 Test 3: Feasibility and Constrained Optimization")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=3)
    
    # Test initial feasibility
    constraint_bounds = constrained_ucb._calculate_constraint_lcb_bounds()
    feasible_arms = constrained_ucb._find_feasible_arms(constraint_bounds)
    
    assert isinstance(feasible_arms, list), "Feasible arms should be list"
    assert len(feasible_arms) > 0, "Should have feasible arms initially"
    print(f"✅ Initial feasibility check: {len(feasible_arms)} feasible arms")
    
    # Test constrained optimization
    selected_prices = constrained_ucb.select_prices()
    if selected_prices:
        selected_price = selected_prices[0]
        selected_arm = constrained_ucb._price_to_arm_id(selected_price)
        assert selected_arm in feasible_arms, "Selected arm should be feasible"
        print("✅ Constrained optimization selects feasible arms")
    
    # Test capacity consumption
    initial_capacity = constrained_ucb.remaining_capacity
    if selected_prices:
        # Simulate successful purchase
        buyer_info = {"round": 0}
        rewards = {0: selected_prices[0]}
        constrained_ucb.update(selected_prices, rewards, buyer_info)
        
        assert constrained_ucb.remaining_capacity == initial_capacity - 1, "Capacity should be consumed"
        print("✅ Capacity consumption works correctly")

def test_constraint_learning():
    """Test that constraint statistics are learned correctly"""
    print("\n🧪 Test 4: Constraint Learning")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=10)
    
    # Test constraint learning with known scenarios
    scenarios = [
        (0.3, 0.5, True),   # price, valuation, purchase
        (0.4, 0.6, True),
        (0.5, 0.3, False),
        (0.3, 0.7, True),   # Same price again
    ]
    
    for round_num, (price, valuation, purchase) in enumerate(scenarios):
        selected_prices = {0: price}
        reward = price if purchase else 0.0
        buyer_info = {"round": round_num}
        constrained_ucb.update(selected_prices, {0: reward}, buyer_info)
        
        # Check constraint statistics
        arm_id = constrained_ucb._price_to_arm_id(price)
        assert constrained_ucb.constraint_counts[arm_id] > 0, "Constraint count should increment"
        # Each production should consume exactly 1 capacity unit
        assert constrained_ucb.constraint_means[arm_id] == 1.0, "Each production should consume 1 capacity"
    
    print("✅ Constraint statistics learned correctly")
    
    # Test constraint statistics retrieval
    constraint_stats = constrained_ucb.get_constraint_statistics()
    assert len(constraint_stats) == len(constrained_ucb.prices), "Statistics for all arms"
    
    for arm_name, stats in constraint_stats.items():
        required_fields = ["arm_id", "price", "constraint_observations", "mean_constraint_cost"]
        for field in required_fields:
            assert field in stats, f"Missing constraint statistic: {field}"
    
    print("✅ Constraint statistics retrieval works")

def test_capacity_management():
    """Test capacity management and tracking"""
    print("\n🧪 Test 5: Capacity Management")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=3)
    
    # Test capacity exhaustion
    production_count = 0
    for round_num in range(6):  # Try more rounds than capacity
        selected_prices = constrained_ucb.select_prices()
        
        if selected_prices:  # If algorithm decided to produce
            production_count += 1
            # Simulate purchase
            price = selected_prices[0]
            buyer_info = {"round": round_num}
            rewards = {0: price}
            constrained_ucb.update(selected_prices, rewards, buyer_info)
        else:
            # No production
            constrained_ucb.update({}, {}, {"round": round_num})
    
    # Should not exceed capacity
    assert production_count <= 3, f"Should not exceed capacity of 3, produced {production_count}"
    assert constrained_ucb.remaining_capacity >= 0, "Remaining capacity should be non-negative"
    print(f"✅ Capacity constraint respected: produced {production_count}/3 times")
    
    # Test capacity statistics
    capacity_stats = constrained_ucb.get_capacity_statistics()
    required_stats = ["total_capacity", "remaining_capacity", "capacity_used", 
                     "production_rounds", "production_rate"]
    for stat in required_stats:
        assert stat in capacity_stats, f"Missing capacity statistic: {stat}"
    
    assert capacity_stats["total_capacity"] == 3, "Total capacity should be 3"
    assert capacity_stats["production_rounds"] == production_count, "Production rounds should match count"
    print("✅ Capacity statistics tracking works")
    
    # Test capacity reset
    constrained_ucb.reset_capacity()
    assert constrained_ucb.remaining_capacity == 3, "Capacity should be reset to full"
    print("✅ Capacity reset works")

def test_integration_with_environment():
    """Test integration with stochastic environment"""
    print("\n🧪 Test 6: Environment Integration")
    print("-" * 50)
    
    # Create environment and both algorithms
    env = create_default_environment()
    ucb1_normal = create_default_ucb1()
    ucb1_constrained = create_default_constrained_ucb1(capacity=20)  # Generous capacity
    
    algorithms = [
        ("UCB1 Normal", ucb1_normal),
        ("UCB1 Constrained", ucb1_constrained)
    ]
    
    for alg_name, algorithm in algorithms:
        env.reset()
        algorithm.reset()
        
        total_reward = 0
        production_rounds = 0
        
        for round_num in range(30):
            selected_prices = algorithm.select_prices()
            
            if selected_prices:  # If algorithm decided to produce
                buyer_info, rewards, done = env.step(selected_prices)
                algorithm.update(selected_prices, rewards, buyer_info)
                total_reward += rewards[0]
                production_rounds += 1
            else:  # No production
                algorithm.update({}, {}, {"round": round_num})
        
        avg_reward = total_reward / max(1, production_rounds)
        production_rate = production_rounds / 30
        
        print(f"✅ {alg_name}: total_reward=${total_reward:.2f}, "
              f"avg_reward=${avg_reward:.3f}, production_rate={production_rate:.3f}")

def test_auction_theory_compliance():
    """Test compliance with auction theory optimization pattern"""
    print("\n🧪 Test 7: Auction Theory Compliance")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=5)
    
    # Give algorithm learning data
    training_data = [
        (0.2, 0.4, True),   # (price, valuation, purchase)
        (0.5, 0.6, True),
        (0.8, 0.7, False),
        (0.3, 0.5, True),
        (0.6, 0.8, True),
        (0.4, 0.3, False),
    ]
    
    for round_num, (price, valuation, purchase) in enumerate(training_data):
        selected_prices = {0: price}
        reward = price if purchase else 0.0
        buyer_info = {"round": round_num}
        constrained_ucb.update(selected_prices, {0: reward}, buyer_info)
    
    # Test auction theory optimization pattern:
    # max f̄^UCB(a) subject to c̄^LCB(a) ≤ remaining_capacity
    
    reward_ucb = constrained_ucb._calculate_reward_ucb_bounds()
    constraint_lcb = constrained_ucb._calculate_constraint_lcb_bounds()
    feasible_arms = constrained_ucb._find_feasible_arms(constraint_lcb)
    
    if feasible_arms:
        # Among feasible arms, should select one with highest reward UCB
        best_feasible_reward = max(reward_ucb[arm] for arm in feasible_arms)
        
        # Simulate selection
        selected_prices = constrained_ucb.select_prices()
        if selected_prices:
            selected_arm = constrained_ucb._price_to_arm_id(selected_prices[0])
            selected_reward_ucb = reward_ucb[selected_arm]
            
            # Should be among feasible arms
            assert selected_arm in feasible_arms, "Selected arm should be feasible"
            
            # Should have high reward UCB (allowing for ties)
            assert selected_reward_ucb >= best_feasible_reward - 1e-6, \
                "Should select arm with high reward UCB among feasible"
            
            print("✅ Follows auction theory optimization: max f̄^UCB(a) s.t. c̄^LCB(a) ≤ capacity")
    
    # Test dual bounds orientation
    for arm_id in range(len(constrained_ucb.prices)):
        if constrained_ucb.arm_counts[arm_id] > 0:  # For arms with data
            # Reward bounds should be optimistic (upper bound)
            mean_reward = constrained_ucb.arm_means[arm_id]
            ucb_reward = reward_ucb[arm_id]
            assert ucb_reward >= mean_reward - 1e-6, "Reward UCB should be optimistic (≥ mean)"
            
            # Constraint bounds should be conservative (lower bound)
            mean_constraint = constrained_ucb.constraint_means[arm_id] 
            lcb_constraint = constraint_lcb[arm_id]
            assert lcb_constraint <= mean_constraint + 1e-6, "Constraint LCB should be conservative (≤ mean)"
    
    print("✅ Dual confidence bounds correctly oriented (optimistic rewards, conservative constraints)")

def test_comparison_vs_basic_ucb1():
    """Compare UCB-Constrained vs basic UCB1"""
    print("\n🧪 Test 8: Comparison vs Basic UCB1")
    print("-" * 50)
    
    # Create both algorithms
    basic_ucb1 = create_default_ucb1()
    constrained_ucb1 = create_default_constrained_ucb1(capacity=25)  # Generous capacity
    
    # Create environment
    env = create_default_environment()
    
    algorithms = [
        ("Basic UCB1", basic_ucb1),
        ("UCB-Constrained", constrained_ucb1)
    ]
    
    results = {}
    
    for alg_name, algorithm in algorithms:
        env.reset()
        algorithm.reset()
        
        total_reward = 0
        production_count = 0
        
        for round_num in range(50):
            # Generate consistent buyer for both algorithms
            valuation = env._sample_valuation()
            
            selected_prices = algorithm.select_prices()
            if selected_prices:
                price = selected_prices[0]
                purchase = valuation >= price
                reward = price if purchase else 0.0
                
                buyer_info = {
                    "valuations": {0: valuation},
                    "purchases": {0: purchase},
                    "round": round_num
                }
                algorithm.update(selected_prices, {0: reward}, buyer_info)
                total_reward += reward
                production_count += 1
            else:
                algorithm.update({}, {}, {"round": round_num})
        
        avg_reward = total_reward / max(1, production_count)
        production_rate = production_count / 50
        
        results[alg_name] = {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "production_rate": production_rate,
            "production_count": production_count
        }
    
    # Display results
    for alg_name, metrics in results.items():
        print(f"📊 {alg_name}:")
        print(f"   Total reward: ${metrics['total_reward']:.2f}")
        print(f"   Average reward: ${metrics['avg_reward']:.3f}")
        print(f"   Production rate: {metrics['production_rate']:.3f}")
        print(f"   Production count: {metrics['production_count']}")
    
    # Basic sanity checks
    for alg_name, metrics in results.items():
        assert metrics["total_reward"] >= 0, f"{alg_name} should have non-negative total reward"
        assert 0 <= metrics["production_rate"] <= 1, f"{alg_name} production rate should be in [0,1]"
    
    # With generous capacity, constrained should perform reasonably
    if results["UCB-Constrained"]["production_count"] > 0:
        performance_ratio = (results["UCB-Constrained"]["total_reward"] / 
                           max(results["Basic UCB1"]["total_reward"], 0.01))
        print(f"📈 Performance ratio (constrained/basic): {performance_ratio:.3f}")
    
    print("✅ Comparison completed successfully")

def test_feasibility_analysis():
    """Test feasibility analysis and bounds comparison"""
    print("\n🧪 Test 9: Feasibility Analysis")
    print("-" * 50)
    
    constrained_ucb = create_default_constrained_ucb1(capacity=4)
    
    # Run some rounds to build up statistics
    scenarios = [
        (0.2, 0.4, True),
        (0.5, 0.7, True),
        (0.8, 0.6, False),
        (0.3, 0.5, True),
        (0.6, 0.8, True),  # This should exhaust capacity
    ]
    
    feasibility_history = []
    
    for round_num, (price, valuation, purchase) in enumerate(scenarios):
        # Check feasibility before selection
        constraint_bounds = constrained_ucb._calculate_constraint_lcb_bounds()
        feasible_arms = constrained_ucb._find_feasible_arms(constraint_bounds)
        feasibility_history.append(len(feasible_arms))
        
        # Force specific price for testing
        if len(feasible_arms) > 0:  # Only if some arms are feasible
            selected_prices = {0: price}
            reward = price if purchase else 0.0
            buyer_info = {"round": round_num}
            constrained_ucb.update(selected_prices, {0: reward}, buyer_info)
        else:
            # No feasible arms
            constrained_ucb.update({}, {}, {"round": round_num})
    
    # Test feasibility analysis
    feasibility_analysis = constrained_ucb.get_feasibility_analysis()
    
    required_fields = ["total_rounds", "rounds_with_feasible_arms", "rounds_infeasible", 
                      "average_feasible_arms", "feasibility_rate"]
    for field in required_fields:
        assert field in feasibility_analysis, f"Missing feasibility analysis field: {field}"
    
    print(f"✅ Feasibility analysis:")
    for key, value in feasibility_analysis.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Test bounds comparison
    bounds_comparison = constrained_ucb.compare_ucb_vs_constraint_bounds()
    assert len(bounds_comparison) == len(constrained_ucb.prices), "Bounds comparison for all arms"
    
    print("✅ Bounds comparison generated for all arms")
    
    # Show top arms with bounds
    sorted_bounds = sorted(bounds_comparison.items(), 
                          key=lambda x: x[1]['reward_ucb'], reverse=True)[:3]
    print("📈 Top 3 arms by reward UCB:")
    for arm_name, data in sorted_bounds:
        feasible = "✅" if data['is_feasible'] else "❌"
        print(f"   {arm_name}: reward_ucb={data['reward_ucb']:.3f}, "
              f"constraint_lcb={data['constraint_lcb']:.3f}, feasible={feasible}")

def run_all_tests():
    """Run all tests for corrected UCB-Constrained algorithm"""
    print("🚀 CORRECTED UCB-CONSTRAINED ALGORITHM TEST SUITE")
    print("🎯 Testing Auction Theory Compliance")
    print("=" * 70)
    
    tests = [
        ("Auction Theory Structure", test_auction_theory_structure),
        ("Dual Confidence Bounds", test_dual_confidence_bounds),
        ("Feasibility & Optimization", test_feasibility_and_optimization),
        ("Constraint Learning", test_constraint_learning),
        ("Capacity Management", test_capacity_management),
        ("Environment Integration", test_integration_with_environment),
        ("Auction Theory Compliance", test_auction_theory_compliance),
        ("Comparison vs Basic UCB1", test_comparison_vs_basic_ucb1),
        ("Feasibility Analysis", test_feasibility_analysis)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            passed += 1
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"🎯 TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Your UCB-Constrained algorithm correctly follows auction theory!")
        print("\n💡 Key achievements:")
        print("   ✅ Follows Multi-Armed Bandit paradigm")
        print("   ✅ Implements dual confidence bounds (f̄^UCB, c̄^LCB)")
        print("   ✅ Performs constrained optimization: max f̄^UCB s.t. c̄^LCB ≤ capacity")
        print("   ✅ Learns constraint statistics correctly")
        print("   ✅ Integrates with stochastic environment")
        print("\n🚀 Next steps:")
        print("   1. Try the demo: python -c \"from algorithms.single_product.constrained_ucb import demo_constrained_ucb1; demo_constrained_ucb1()\"")
        print("   2. Implement your Requirement 1 experiments")
        print("   3. Compare auction-theory approach vs heuristic approaches")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        print("💡 The algorithm should follow the auction theory paradigm:")
        print("   - Upper confidence bounds for rewards (optimistic)")
        print("   - Lower confidence bounds for constraints (conservative)")
        print("   - Constrained optimization over feasible arms")
    
    return failed == 0

def main():
    """Run all tests"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()