from env.inventory_env import InventoryEnv, CostParams
from agents.restock_recommender_agent import fixed_order_policy, order_up_to_S, ForecastAwarePolicy
from agents.demand_forecast_agent import MovingAverageForecaster, EMAForecaster
from utils.log_utils import save_simulation_log

def run_policy(policy_fn, policy_name, episodes=1, horizon=52):
    env = InventoryEnv(
        n_warehouses=2,
        horizon=horizon,
        cost_params=CostParams(holding_cost=1.0, stockout_cost=5.0, order_cost=0.5)
    )
    logs = []
    total_costs, holding_costs, stockout_costs, order_costs, service_levels = [], [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0

        while not done:
            # If this is a ForecastAwarePolicy instance, update its forecaster with last demand
            if hasattr(policy_fn, "forecaster") and (ep > 0 or env.t > 0):
                policy_fn.forecaster.update(env._last_demand)

            action = policy_fn(obs) if callable(policy_fn) else policy_fn(obs)  # works for both lambdas & objects
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            total_costs.append(-reward)
            holding_costs.append(info["holding_cost"])
            stockout_costs.append(info["stockout_cost"])
            order_costs.append(info["order_cost"])
            service_levels.append(info["service_level"])

            logs.append({
                "step": env.t,
                "action": action.tolist(),
                "reward": reward,
                **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()}
            })

        print(f"[{policy_name}] Episode {ep+1} total reward: {total_reward:.2f}")

    print(f"\nSummary for {policy_name}:")
    print(f"  Avg Total Cost:   {sum(total_costs)/len(total_costs):.2f}")
    print(f"    Avg Holding:    {sum(holding_costs)/len(holding_costs):.2f}")
    print(f"    Avg Stockout:   {sum(stockout_costs)/len(stockout_costs):.2f}")
    print(f"    Avg Order Cost: {sum(order_costs)/len(order_costs):.2f}")
    print(f"  Avg Service Level: {sum(service_levels)/len(service_levels):.2%}\n")

    save_simulation_log(logs, policy_name)

if __name__ == "__main__":
    print("Running Fixed Order Policy...")
    run_policy(lambda obs: fixed_order_policy(obs, fixed=20), "fixed_order", episodes=10, horizon=52)

    print("Running Order-up-to-S Policy...")
    run_policy(lambda obs: order_up_to_S(obs, S=80), "order_up_to_S", episodes=10, horizon=52)

    print("Running Forecast-Aware Order-up-to-S Policy...")
    fa_forecaster = MovingAverageForecaster(window=7, n_warehouses=2)   # a bit smoother window
    forecast_policy = ForecastAwarePolicy(
        base_policy=order_up_to_S,
        forecaster=fa_forecaster,
        target_S=80,
        adjustment_factor=1.2
    )
    print("Running Forecast-Aware Order-up-to-S Policy...")
    fa_forecaster = EMAForecaster(alpha=0.4, n_warehouses=2)   
    forecast_policy = ForecastAwarePolicy(
    base_policy=order_up_to_S,
    forecaster=fa_forecaster,
    target_S=80,
    adjustment_factor=1.2
    )
    run_policy(forecast_policy, "forecast_order_up_to_S", episodes=10, horizon=52)
