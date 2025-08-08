import numpy as np
import gymnasium as gym
from gymnasium import spaces
from custom_tools.cost_simulation_tool import CostParams, compute_costs

class InventoryEnv(gym.Env):
    """
    Multi-warehouse inventory environment.
    - Discrete order amounts per warehouse
    - Seasonal + noisy demand generated inside the env (no external data needed)
    - Lead times via simple order pipeline
    - Reward = negative total cost (holding + stockout + order [+ transfer stub])
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_warehouses: int = 2,
        horizon: int = 52,
        action_levels=(0, 10, 20, 30),
        max_capacity: int = 200,
        lead_times=(1, 1),
        demand_mu=(30, 25),
        demand_season_amp=(10, 8),
        rng_seed: int = 0,
        cost_params: CostParams | None = None,
        allow_transfers: bool = False,  # transfers come later
    ):
        super().__init__()
        # config
        self.W = int(n_warehouses)
        self.H = int(horizon)
        self.action_levels = np.array(action_levels, dtype=np.int32)
        self.A = len(self.action_levels)
        self.max_capacity = int(max_capacity)
        self.lead_times = np.array(lead_times, dtype=np.int32)
        self.demand_mu = np.array(demand_mu, dtype=float)
        self.demand_season_amp = np.array(demand_season_amp, dtype=float)
        self.allow_transfers = allow_transfers
        self.cost_params = cost_params or CostParams()
        self.rng = np.random.default_rng(rng_seed)

        # Observation: [stock(W), forecast(W), in_transit(W), time_idx]
        high = np.array(
            [self.max_capacity]*self.W +     # stock
            [max(1, int(max(self.demand_mu) + 3*max(self.demand_season_amp)))]*self.W +  # forecast cap
            [self.max_capacity]*self.W +     # in_transit
            [self.H],                        # time
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        # Action: choose one discrete order level per warehouse
        self.action_space = spaces.MultiDiscrete([self.A]*self.W)

        self._reset_internal()

    # ---------- Core RL API ----------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._observe(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (self.W,), f"Expected action shape {(self.W,)}, got {action.shape}"
        # map action indices -> order quantities
        orders = self.action_levels[action].astype(np.int32)

        # arrivals first (deliveries whose time has come)
        total_arrivals = np.zeros(self.W, dtype=np.int32)
        for w in range(self.W):
            arrivals = 0
            keep = []
            for arrive_t, qty in self.pipeline[w]:
                if arrive_t == self.t:
                    arrivals += qty
                else:
                    keep.append((arrive_t, qty))
            self.pipeline[w] = keep
            total_arrivals[w] = arrivals
        self.stock = np.minimum(self.stock + total_arrivals, self.max_capacity)

        # capacity clamp on new orders
        cap_gap = self.max_capacity - self.stock
        orders = np.minimum(orders, cap_gap)

        # register new orders into pipeline
        for w in range(self.W):
            q = int(orders[w])
            if q > 0:
                arrive_t = self.t + int(self.lead_times[w])
                self.pipeline[w].append((arrive_t, q))
        self.in_transit = np.array([sum(q for _, q in self.pipeline[w]) for w in range(self.W)], dtype=np.int32)

        # sample demand and fulfill
        demand = self._sample_demand(self.t)
        fulfilled = np.minimum(self.stock, demand)
        stockouts_units = demand - fulfilled
        self.stock = self.stock - fulfilled  # post-demand stock

        # costs
        holding_units = float(np.sum(self.stock))
        stockout_units = float(np.sum(stockouts_units))
        ordered_units = float(np.sum(orders))
        transferred_units = 0.0  # transfers not implemented on Day 1

        costs = compute_costs(
            holding_units=holding_units,
            stockout_units=stockout_units,
            ordered_units=ordered_units,
            transferred_units=transferred_units,
            p=self.cost_params,
        )
        reward = -float(costs["total"])

        # log, advance time
        self.t += 1
        terminated = self.t >= self.H
        truncated = False

        info = {
            "holding_cost": costs["holding"],
            "stockout_cost": costs["stockout"],
            "order_cost": costs["order"],
            "transfer_cost": costs["transfer"],
            "demand": demand,
            "fulfilled": fulfilled,
            "orders": orders,
            "service_level": 1.0 - (np.sum(stockouts_units) / max(1, np.sum(demand))),
        }
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        print(f"t={self.t} stock={self.stock} in_transit={self.in_transit}")

    # ---------- Internals ----------

    def _reset_internal(self):
        self.t = 0
        # start with some stock
        self.stock = np.full(self.W, self.max_capacity // 4, dtype=np.int32)
        self.in_transit = np.zeros(self.W, dtype=np.int32)
        self.pipeline = {w: [] for w in range(self.W)}  # list[(arrive_t, qty)]
        self._last_demand = np.zeros(self.W, dtype=np.int32)

    def _observe(self):
        forecast = self._forecast(self.t)
        obs = np.concatenate([self.stock, forecast, self.in_transit, np.array([self.t])]).astype(np.float32)
        return obs

    def _forecast(self, t):
        # Naive Day 1: use last demand as the forecast; at t=0 use seasonal mean
        if self.t == 0:
            return self._seasonal_mean(t).astype(np.float32)
        return self._last_demand.astype(np.float32)

    def _seasonal_mean(self, t):
        # seasonal multiplier in [0.5, 1.5] over 52-step cycle
        season = 1.0 + 0.5*np.sin(2*np.pi*(t % 52)/52.0)
        return self.demand_mu * season

    def _sample_demand(self, t):
        # Poisson around seasonal mean + small Gaussian perturbation scaled by amp
        mean = self._seasonal_mean(t) + self.rng.normal(0, 0.1, size=self.W) * self.demand_season_amp
        mean = np.clip(mean, 0.1, None)
        demand = self.rng.poisson(mean).astype(np.int32)
        self._last_demand = demand.copy()
        return demand

if __name__ == "__main__":
    from custom_tools.cost_simulation_tool import CostParams

    env = InventoryEnv(
        n_warehouses=2,
        horizon=10,  # shorter for quick test
        cost_params=CostParams(holding_cost=1.0, stockout_cost=5.0, order_cost=0.5)
    )

    obs, _ = env.reset(seed=42)
    done = False
    total_reward = 0.0

    print("Starting random rollout...")
    while not done:
        action = env.action_space.sample()  # pick a random order amount
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print(f"Step {env.t}: action={action}, reward={reward:.2f}, info={info}")