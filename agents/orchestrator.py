# agents/orchestrator.py
import numpy as np
from typing import Dict, Callable, Optional, Tuple, Any, List
from agents.policy_selector import UCBBandit
from agents.demand_forecast_agent import EMAForecaster

class SafetyFallback:
    """Safety mechanism to override RL decisions in critical situations."""
    
    def __init__(self, 
                 target_service: float = 0.92,
                 safety_stock_mult: float = 1.5,
                 action_levels: tuple = (0, 10, 20, 40, 60, 80)):
        """
        Initialize safety fallback.
        
        Args:
            target_service: Target service level
            safety_stock_mult: Multiplier for safety stock
            action_levels: Tuple of possible action values (matching env)
        """
        self.target_service = target_service
        self.safety_stock_mult = safety_stock_mult
        self.action_levels = np.array(action_levels, dtype=np.int32)
        
    def should_trigger(self, obs: np.ndarray, n_warehouses: int = 2) -> bool:
        """
        Check if safety fallback should be triggered.
        
        Observation format: [stock(W), forecast(W), in_transit(W), time_idx]
        """
        current_stock = obs[:n_warehouses]
        
        # Get forecast from observation
        if len(obs) > n_warehouses * 2:
            forecast = obs[n_warehouses:n_warehouses*2]
            recent_demand = np.mean(forecast) if np.any(forecast > 0) else 30
        else:
            recent_demand = 30  # Default estimate based on demand_mu
        
        # Trigger if any warehouse is critically low
        safety_stock = recent_demand * self.safety_stock_mult
        
        for stock in current_stock:
            if stock < safety_stock:
                return True
        return False
        
    def get_action(self, obs: np.ndarray, n_warehouses: int = 2) -> np.ndarray:
        """
        Get safe fallback action (order-up-to-S policy).
        Returns action INDICES for the environment.
        """
        current_stock = obs[:n_warehouses]
        
        # Get forecast from observation
        if len(obs) > n_warehouses * 2:
            forecast = obs[n_warehouses:n_warehouses*2]
            recent_demand = np.mean(forecast) if np.any(forecast > 0) else 30
        else:
            recent_demand = 30
        
        # Order up to safety stock level
        target_stock = recent_demand * 2.0  # 2x average demand
        
        actions = []
        for stock in current_stock:
            order_qty = max(0, target_stock - stock)
            order_qty = min(order_qty, 80)  # Cap at max action level
            
            # Find closest action level INDEX
            idx = np.argmin(np.abs(self.action_levels - order_qty))
            actions.append(idx)
            
        return np.array(actions, dtype=np.int64)


class RLPolicyWrapper:
    """
    Wrapper for RL policies to handle their specific output formats and normalization.
    """
    def __init__(self, model, algo_type: str, n_warehouses: int = 2, vec_norm=None):
        """
        Args:
            model: Loaded SB3 model (DQN or PPO)
            algo_type: 'dqn' or 'ppo'
            n_warehouses: Number of warehouses
            vec_norm: VecNormalize wrapper for observation normalization
        """
        self.model = model
        self.algo_type = algo_type.lower()
        self.n_warehouses = n_warehouses
        self.vec_norm = vec_norm
        
        # For DQN with MultiDiscreteToDiscreteWrapper
        if self.algo_type == 'dqn':
            self.base = 6  # Number of action options per warehouse (len(action_levels))
            
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from RL model.
        Returns action indices for the environment.
        """
        # Normalize observation if vec_norm is available
        if self.vec_norm is not None:
            # VecNormalize expects batch dimension
            norm_obs = self.vec_norm.normalize_obs(obs.reshape(1, -1))
            obs_to_use = norm_obs
        else:
            # Add batch dimension
            obs_to_use = obs.reshape(1, -1)
            
        # Get action from model
        action, _ = self.model.predict(obs_to_use, deterministic=True)
        
        if self.algo_type == 'dqn':
            # DQN returns a flat integer, convert to multi-discrete indices
            # Using same logic as MultiDiscreteToDiscreteWrapper.action()
            flat_action = int(action[0])
            indices = []
            a = flat_action
            for _ in range(self.n_warehouses):
                indices.append(a % self.base)
                a //= self.base
            return np.array(indices, dtype=np.int64)
        else:
            # PPO returns multi-discrete indices directly
            return action.flatten().astype(np.int64)


class Orchestrator:
    """Orchestrates between multiple policies with safety fallback."""
    
    def __init__(self,
                 action_levels: tuple = (0, 10, 20, 40, 60, 80),
                 n_warehouses: int = 2,
                 target_service: float = 0.92,
                 below_target_mult: float = 8.0):
        """
        Initialize orchestrator.
        
        Args:
            action_levels: Tuple of possible action values (matching env)
            n_warehouses: Number of warehouses
            target_service: Target service level for reward shaping
            below_target_mult: Multiplier for below-target penalty
        """
        self.action_levels = np.array(action_levels, dtype=np.int32)
        self.n_warehouses = n_warehouses
        self.target_service = target_service
        self.below_target_mult = below_target_mult
        
        # Initialize components
        self.policies = {}
        self.bandit = None
        self.forecaster = EMAForecaster(alpha=0.3, n_warehouses=n_warehouses)
        self.fallback = SafetyFallback(
            target_service=target_service,
            action_levels=action_levels
        )
        
        # Tracking
        self.episode_costs = []
        self.episode_services = []
        self.episode_meta = []
        self.current_episode_cost = 0
        self.current_episode_services = []
        
    def add_policy(self, name: str, policy_fn: Callable):
        """
        Add a policy to the orchestrator.
        
        Args:
            name: Policy name
            policy_fn: Function that takes obs and returns action indices
        """
        self.policies[name] = policy_fn
        
    def add_rl_policy(self, name: str, model_path: str, algo_type: str):
        """
        Add an RL policy with proper wrapper and normalization.
        
        Args:
            name: Policy name
            model_path: Path to saved model
            algo_type: 'dqn' or 'ppo'
        """
        import os
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from env.wrappers import MultiDiscreteToDiscreteWrapper
        
        # Load model
        if algo_type.lower() == 'dqn':
            model = DQN.load(model_path)
        else:
            model = PPO.load(model_path)
        
        # Try to load VecNormalize stats
        vec_norm = None
        vecnorm_path = os.path.join(
            os.path.dirname(model_path),
            f"vecnorm_{algo_type.lower()}_W{self.n_warehouses}.pkl"
        )
        
        if os.path.exists(vecnorm_path):
            # Create dummy env to load normalization
            from env.inventory_env import InventoryEnv
            
            def make_env():
                env = InventoryEnv(
                    n_warehouses=self.n_warehouses,
                    action_levels=tuple(self.action_levels),
                    target_service=self.target_service,
                    below_target_mult=self.below_target_mult
                )
                if algo_type.lower() == 'dqn':
                    env = MultiDiscreteToDiscreteWrapper(env)
                return env
            
            vec_env = DummyVecEnv([make_env])
            vec_norm = VecNormalize.load(vecnorm_path, vec_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
        
        # Create wrapper
        wrapper = RLPolicyWrapper(model, algo_type, self.n_warehouses, vec_norm)
        self.policies[name] = wrapper.predict
        
    def initialize_bandit(self):
        """Initialize the bandit with available policies."""
        if not self.policies:
            raise ValueError("No policies added to orchestrator")
        
        self.bandit = UCBBandit(list(self.policies.keys()))
        
    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        Select and execute action for current state.
        
        Args:
            obs: Environment observation
            
        Returns:
            actions: Array of action INDICES for the environment
        """
        # Check if we need safety fallback
        use_fallback = self.fallback.should_trigger(obs, self.n_warehouses)
        
        if use_fallback:
            # Get fallback action indices
            action_indices = self.fallback.get_action(obs, self.n_warehouses)
            
            meta = {
                "policy": "safety_fallback",
                "fallback_used": True,
                "actions": action_indices.tolist()
            }
        else:
            # Select policy via bandit
            if self.bandit is None:
                self.initialize_bandit()
                
            policy_name = self.bandit.select()
            policy_fn = self.policies[policy_name]
            
            # Get action indices from policy
            try:
                action_indices = policy_fn(obs)
            except Exception as e:
                print(f"Error in policy {policy_name}: {e}")
                # Fallback to safe action
                action_indices = self.fallback.get_action(obs, self.n_warehouses)
                policy_name = "safety_fallback (error)"
            
            # Ensure correct shape and type
            action_indices = np.array(action_indices, dtype=np.int64)
            if action_indices.shape != (self.n_warehouses,):
                action_indices = action_indices.flatten()[:self.n_warehouses]
            
            # Validate indices are in valid range
            action_indices = np.clip(action_indices, 0, len(self.action_levels) - 1)
            
            meta = {
                "policy": policy_name,
                "fallback_used": False,
                "actions": action_indices.tolist(),
                "bandit_stats": self.bandit.get_stats() if self.bandit else {}
            }
            
        self.episode_meta.append(meta)
        return action_indices
        
    def update_step_metrics(self, info: Dict):
        """
        Update metrics after environment step.
        
        Args:
            info: Info dict from environment step
        """
        # Track costs
        step_cost = (
            info.get("holding_cost", 0) +
            info.get("stockout_cost", 0) +
            info.get("order_cost", 0) +
            info.get("transfer_cost", 0)
        )
        self.current_episode_cost += step_cost
        
        # Track service level
        self.current_episode_services.append(info.get("service_level", 0))
        
        # Update forecaster if demand is available
        if "demand" in info:
            self.forecaster.update(info["demand"])
            
    def end_episode(self):
        """End current episode and update bandit."""
        if self.current_episode_services:
            avg_service = np.mean(self.current_episode_services)
        else:
            avg_service = 0
            
        # Update bandit with episode results
        if self.bandit is not None:
            # Use negative cost as reward (lower cost = higher reward)
            # Add bonus for meeting service target
            service_bonus = 1000 if avg_service >= self.target_service else 0
            reward = -self.current_episode_cost + service_bonus
            
            self.bandit.update(reward)
            
        # Track episode metrics
        self.episode_costs.append(self.current_episode_cost)
        self.episode_services.append(avg_service)
        
        # Reset for next episode
        self.current_episode_cost = 0
        self.current_episode_services = []
        self.episode_meta = []
        
    def get_performance_summary(self) -> Dict:
        """Get orchestrator performance statistics."""
        if not self.episode_costs:
            return {"message": "No episodes completed yet"}
            
        return {
            "episodes_run": len(self.episode_costs),
            "mean_cost": float(np.mean(self.episode_costs)),
            "std_cost": float(np.std(self.episode_costs)),
            "best_cost": float(min(self.episode_costs)),
            "worst_cost": float(max(self.episode_costs)),
            "mean_service": float(np.mean(self.episode_services)),
            "std_service": float(np.std(self.episode_services)),
            "bandit_stats": self.bandit.get_stats() if self.bandit else {},
            "policies_used": list(self.policies.keys())
        }
        
    def reset(self):
        """Reset orchestrator state."""
        self.episode_costs = []
        self.episode_services = []
        self.episode_meta = []
        self.current_episode_cost = 0
        self.current_episode_services = []
        if self.bandit:
            self.bandit.reset()