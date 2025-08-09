import numpy as np
from collections import deque

class MovingAverageForecaster:
    """(kept if you still want MA)"""
    def __init__(self, window=5, n_warehouses=2):
        self.window = window
        self.n_warehouses = n_warehouses
        self.buffers = [deque(maxlen=window) for _ in range(n_warehouses)]
    def update(self, demand_vec):
        for w in range(self.n_warehouses):
            self.buffers[w].append(int(demand_vec[w]))
    def predict(self):
        if any(len(b) == 0 for b in self.buffers):
            return np.zeros(self.n_warehouses, dtype=np.float32)
        return np.array([np.mean(b) for b in self.buffers], dtype=np.float32)

class EMAForecaster:
    """Exponential moving average forecaster per warehouse."""
    def __init__(self, alpha=0.4, n_warehouses=2):
        self.alpha = float(alpha)
        self.n_warehouses = n_warehouses
        self.ema = np.zeros(n_warehouses, dtype=float)
        self.inited = False
    def update(self, demand_vec):
        d = np.array(demand_vec, dtype=float)
        if not self.inited:
            self.ema = d
            self.inited = True
        else:
            self.ema = self.alpha * d + (1 - self.alpha) * self.ema
    def predict(self):
        return self.ema.astype(np.float32) if self.inited else np.zeros(self.n_warehouses, dtype=np.float32)
