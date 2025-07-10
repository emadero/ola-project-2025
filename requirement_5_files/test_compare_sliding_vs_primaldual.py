from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment
from requirement_5_files.sliding_window_cusum import SlidingWindowCUCB
from algorithms.multiple_products.primal_dual import PrimalDualMultipleProducts

def test_compare():
    prices = [0.2, 0.4, 0.6, 0.8]
    config = dict(n_products=3, prices=prices, production_capacity=60, total_rounds=20, n_intervals=2)

    env_sw = MultiProductPiecewiseStationaryEnvironment(**config)
    agent_sw = SlidingWindowCUCB(n_products=3, prices=prices, window_size=10)
    env_sw.reset()
    while True:
        sel = agent_sw.select_prices()
        _, rew, done = env_sw.step(sel)
        if isinstance(rew, list):
            rew = {p: rew[p] for p in range(len(rew))}
        agent_sw.update(sel, rew)
        if done:
            break

    env_pd = MultiProductPiecewiseStationaryEnvironment(**config)
    agent_pd = PrimalDualMultipleProducts(prices, 3, 60, 20, 0.01)
    history = agent_pd.run(env_pd)

    assert len(history["revenues"]) == config["total_rounds"]
   

print(" Test passed: SlidingWindowCUCB and PrimalDual ran successfully.")