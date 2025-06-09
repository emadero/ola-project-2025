#!/usr/bin/env python3
"""
Test script for the highly non-stationary multi-product environment
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from environments.multi_non_stationary import MultiProductHighlyNonStationaryEnvironment
    print("✅ Successfully imported MultiProductHighlyNonStationaryEnvironment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure environments/multi_non_stationary.py exists and is implemented correctly")
    sys.exit(1)


def test_basic_functionality():
    print("\n🧪 Testing Basic Functionality")
    print("-" * 40)

    env = MultiProductHighlyNonStationaryEnvironment(
        n_products=3,
        prices=[0.2, 0.3, 0.4, 0.5, 0.6],
        production_capacity=10,
        total_rounds=100,
        random_seed=42
    )
    print("✅ Environment created successfully")

    state = env.reset()
    print(f"✅ Environment reset: {state}")

    buyer_info, rewards, done = env.step({0: 0.3, 1: 0.4, 2: 0.5})
    print("✅ Step executed")
    print(f"   Buyer valuations: {buyer_info['valuations']}")
    print(f"   Purchases: {buyer_info['purchases']}")
    print(f"   Rewards: {rewards}")
    print(f"   Done: {done}")


def test_dynamic_behavior():
    print("\n📈 Testing Dynamic Behavior Over Time")
    print("-" * 40)

    env = MultiProductHighlyNonStationaryEnvironment(
        n_products=2,
        prices=[0.3, 0.4, 0.5],
        production_capacity=150,
        total_rounds=30,
        random_seed=123
    )
    env.reset()

    for t in range(30):
        selected_prices = {0: 0.4, 1: 0.4}
        buyer_info, rewards, done = env.step(selected_prices)
        
        if done:
            print(f"🔚 Simulation ended at round {t + 1} (inventory or rounds exhausted)")
            break
            
        print(f"Round {t + 1}")
        print(f"  Valuations: {buyer_info['valuations']}")
        print(f"  Purchases: {buyer_info['purchases']}")
        print(f"  Rewards: {rewards}")
        print()


def test_exhaustion():
    print("\n🛑 Testing Capacity Exhaustion")
    print("-" * 40)

    env = MultiProductHighlyNonStationaryEnvironment(
        n_products=2,
        prices=[0.2, 0.3],
        production_capacity=3,
        total_rounds=20,
        random_seed=1
    )
    env.reset()

    for i in range(10):
        _, _, done = env.step({0: 0.2, 1: 0.2})
        if done:
            print(f"✅ Environment stopped at round {i + 1} due to inventory exhaustion")
            break
    else:
        print("❌ Environment did not stop as expected")


def main():
    print("🎮 Testing Highly Non-Stationary Multi-Product Environment")
    print("=" * 60)

    test_basic_functionality()
    test_dynamic_behavior()
    test_exhaustion()

    print("\n🎉 All tests completed!")
    print("\n💡 You can now use this environment in your experiments")


if __name__ == "__main__":
    main()
