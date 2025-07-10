from environments.slightly_ns import MultiProductPiecewiseStationaryEnvironment
from requirement_5_files.sliding_window_cusum import SlidingWindowCUCB


def test_sw_run():
    prices = [0.2, 0.4, 0.6, 0.8]
    config = dict(n_products=3, prices=prices, production_capacity=60, total_rounds=20, n_intervals=2)

    env = MultiProductPiecewiseStationaryEnvironment(**config)
    agent = SlidingWindowCUCB(n_products=3, prices=prices, window_size=10)
    env.reset()
    total = 0
    while True:
        sel = agent.select_prices()
        _, rew, done = env.step(sel)
        if isinstance(rew, list):
            rew = {p: rew[p] for p in range(len(rew))}
        agent.update(sel, rew)
        total += sum(rew.values())
        if done:
            break
    assert total > 0  

print("SlidingWindowCUCB test passed.")