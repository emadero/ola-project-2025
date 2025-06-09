#!/usr/bin/env python3
"""
Demo for Multi-Product Highly Non-Stationary Environment
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from environments.multi_non_stationary import MultiProductHighlyNonStationaryEnvironment
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def demo_environment():
    print("🎬 Demo: Multi-Product Highly Non-Stationary Environment")
    print("=" * 50)

    # Setup
    env = MultiProductHighlyNonStationaryEnvironment(
        n_products=3,
        prices=[0.2, 0.3, 0.4, 0.5, 0.6],
        production_capacity=10,
        total_rounds=5,
        random_seed=42
    )

    env.reset()

    # Prices fixed for demo
    fixed_prices = {0: 0.6, 1: 0.7, 2: 0.8}

    # Run 5 demo rounds
    for round_num in range(5):
        buyer_info, rewards, done = env.step(fixed_prices)

        print(f"\nRound {round_num + 1}:")
        print(f"  Valuations: {env.get_buyer_valuations()}")
        print(f"  Purchases:  {buyer_info['purchases']}")
        print(f"  Rewards:    {rewards}")

        if done:
            print("\n🔚 Environment finished.")
            break

if __name__ == "__main__":
    demo_environment()
