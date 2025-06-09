#!/usr/bin/env python3
"""
Test script for the Multi-Product Stochastic Environment
Run this to verify that the multi-product environment behaves as expected
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from environments.multi_stochastic import MultiProductStochasticEnvironment
    print("✅ Successfully imported MultiProductStochasticEnvironment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure environments/multi_stochastic.py exists and is correctly written")
    sys.exit(1)


def test_multi_basic_functionality():
    """Test basic functionality of the multi-product environment"""
    print("\n🧪 Testing Basic Functionality")
    print("-" * 40)

    env = MultiProductStochasticEnvironment(
        n_products=3,
        prices=[0.2, 0.4, 0.6],
        production_capacity=10,
        total_rounds=100,
        valuation_distribution="uniform",
        valuation_params={"low": 0.0, "high": 1.0},
        random_seed=42
    )

    print("✅ Environment created successfully")
    state = env.reset()
    print("✅ Environment reset")
    print(f"   State: {state}")

    selected_prices = {0: 0.3, 1: 0.4, 2: 0.5}
    buyer_info, rewards, done = env.step(selected_prices)

    print("✅ Step executed")
    print(f"   Buyer valuations: {buyer_info['valuations']}")
    print(f"   Purchases: {buyer_info['purchases']}")
    print(f"   Rewards: {rewards}")
    print(f"   Done: {done}")


def test_multi_multiple_rounds():
    """Run multiple rounds with random fixed pricing"""
    print("\n🔁 Testing Multiple Rounds Execution")
    print("-" * 40)

    env = MultiProductStochasticEnvironment(
        n_products=2,
        prices=[0.1, 0.2, 0.3, 0.4, 0.5],
        production_capacity=6,
        total_rounds=10,
        random_seed=123
    )
    env.reset()

    total_revenue = 0
    for t in range(10):
        prices = {0: 0.3, 1: 0.4}
        buyer_info, rewards, done = env.step(prices)
        round_revenue = sum(rewards.values())
        total_revenue += round_revenue

        print(f"Round {t + 1}:")
        print(f"  Valuations: {buyer_info['valuations']}")
        print(f"  Purchases: {buyer_info['purchases']}")
        print(f"  Rewards: {rewards}")
        print(f"  Total revenue so far: {total_revenue:.2f}")
        if done:
            print("🔚 Simulation ended (inventory or rounds exhausted)")
            break


def test_multi_edge_cases():
    """Check how the environment handles edge cases"""
    print("\n🧪 Testing Edge Cases")
    print("-" * 40)

    try:
        env = MultiProductStochasticEnvironment(
            n_products=2,
            prices=[0.5],
            production_capacity=1,
            total_rounds=1,
            valuation_distribution="uniform",
            valuation_params={"low": 0.0, "high": 1.0},
            random_seed=42
        )
        env.reset()
        buyer_info, rewards, done = env.step({0: 0.5, 1: 0.5})
        print("✅ Step with minimal config ran successfully")
        print(f"   Buyer valuations: {buyer_info['valuations']}")
        print(f"   Purchases: {buyer_info['purchases']}")
        print(f"   Rewards: {rewards}")
    except Exception as e:
        print(f"❌ Error during edge case test: {e}")


def main():
    print("🎮 Testing Multi-Product Stochastic Environment")
    print("=" * 60)

    test_multi_basic_functionality()
    test_multi_multiple_rounds()
    test_multi_edge_cases()

    print("\n🎉 All tests completed!")
    print("\n💡 Tip: You can import and run the environment in your experiment like this:")
    print("   from environments.multi_stochastic import MultiProductStochasticEnvironment")


if __name__ == "__main__":
    main()
