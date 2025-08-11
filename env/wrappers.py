# env > wrappers.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    """
    Flattens a MultiDiscrete([A, A, ..., A]) action into a single Discrete(A**W).
    DQN can then act on Discrete; we map back to the original MultiDiscrete.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete), \
            "Wrapper expects a MultiDiscrete action space"
        self.nvec = np.array(env.action_space.nvec, dtype=int)   # e.g., [A, A, ...] length W
        self.W = len(self.nvec)
        # only support all-equal for simplicity, but you can generalize
        self.radices = self.nvec.copy()
        self.base = int(self.radices[0])
        assert np.all(self.radices == self.base), "Currently supports equal sizes per dimension"
        self.action_space = spaces.Discrete(int(self.base ** self.W))

    def action(self, act: int):
        # Convert integer -> base-A digits of length W (little-endian)
        a = int(act)
        idxs = []
        for _ in range(self.W):
            idxs.append(a % self.base)
            a //= self.base
        idxs = idxs  # already little-endian corresponds to warehouses 0..W-1
        return np.array(idxs, dtype=np.int64)

    def reverse_action(self, action_vec):
        # Optional: MultiDiscrete -> flat integer (useful for logging)
        a = 0
        mult = 1
        for i in range(self.W):
            a += int(action_vec[i]) * mult
            mult *= self.base
        return a
