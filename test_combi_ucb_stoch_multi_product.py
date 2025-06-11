from environments.multi_stochastic import MultiProductStochasticEnvironment
from algorithms.multiple_products.combinatorial_ucb import CombinatorialUCB1Algorithm

def main():
    print("🎯 Testing Combinatorial UCB1 with Multi-Product Stochastic Environment")
    print("=" * 60)

    env = MultiProductStochasticEnvironment(
        n_products=3,
        prices=[0.2, 0.3, 0.4, 0.5, 0.6],
        production_capacity=50,
        total_rounds=30,
        random_seed=42
    )

    algo = CombinatorialUCB1Algorithm(
        n_products=3,
        prices=[0.2, 0.3, 0.4, 0.5, 0.6]
    )

    env.reset()

    total_revenue = 0
    for t in range(30):
        prices = algo.select_prices()
        buyer_info, rewards, done = env.step(prices)
        algo.update(prices, rewards)

        round_revenue = sum(rewards.values())
        total_revenue += round_revenue

        print(f"\nRound {t + 1}")
        print(f"  Selected prices: {prices}")
        print(f"  Buyer valuations: {buyer_info['valuations']}")
        print(f"  Purchases: {buyer_info['purchases']}")
        print(f"  Rewards: {rewards}")
        print(f"  Revenue this round: {round_revenue:.2f}")

        if done:
            print("\n🛑 Environment finished.")
            break

    print(f"\n💰 Total Revenue: {total_revenue:.2f}")

if __name__ == "__main__":
    main()
