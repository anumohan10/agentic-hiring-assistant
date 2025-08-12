# agents/restock_recommender_agent.py
import numpy as np
from agents.demand_forecast_agent import MovingAverageForecaster, EMAForecaster

class ForecastAwarePolicy:
    def __init__(self, base_policy, forecaster, target_S=80, action_levels=(0,10,20,30),
                 adjustment_factor=1.2, cap_low=-20, cap_high=40):
        self.base_policy = base_policy
        self.forecaster = forecaster
        self.target_S = target_S
        self.action_levels = action_levels
        self.adjustment_factor = adjustment_factor
        self.cap_low = cap_low     # how far below S we allow
        self.cap_high = cap_high   # how far above S we allow

    def __call__(self, state):
        W = self.forecaster.n_warehouses
        stock = state[:W]
        forecast = self.forecaster.predict()

        centered = forecast - np.mean(forecast)
        adjusted_target = self.target_S + self.adjustment_factor * centered
        # cap around S so we don’t overshoot absurdly
        adjusted_target = np.clip(adjusted_target,
                                  self.target_S + self.cap_low,
                                  self.target_S + self.cap_high)

        gap = np.maximum(0, adjusted_target - stock)
        idxs = [int(np.argmin([abs(a - gap[w]) for a in self.action_levels])) for w in range(W)]
        return np.array(idxs, dtype=np.int64)

def fixed_order_policy(state, action_levels=(0,10,20,30), fixed=20, n_warehouses=2):
    """Always order the same fixed quantity for each warehouse."""
    idx = int(np.argmin([abs(a - fixed) for a in action_levels]))
    return np.array([idx] * n_warehouses, dtype=np.int64)

def order_up_to_S(state, S=80, action_levels=(0,10,20,30), n_warehouses=2):
    """Order enough units to bring each warehouse's stock up to target level S."""
    stock = state[:n_warehouses]
    gap = np.maximum(0, S - stock)
    idxs = [int(np.argmin([abs(a - gap[w]) for a in action_levels])) for w in range(n_warehouses)]
    return np.array(idxs, dtype=np.int64)
