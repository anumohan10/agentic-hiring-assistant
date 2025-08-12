# agents/rl_agent.py 
import numpy as np
import os
from typing import Optional

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. RL features will be limited.")

class RLInventoryAgent:
    """RL-based inventory agent using trained models with enhanced debugging."""
    
    def __init__(self, model_path: str, use_fallback: bool = False, n_warehouses: int = 2):
        """
        Initialize RL agent.
        
        Args:
            model_path: Path to trained model
            use_fallback: Whether to use fallback policy for safety
            n_warehouses: Number of warehouses
        """
        self.model_path = model_path
        self.use_fallback = use_fallback
        self.n_warehouses = n_warehouses
        self.model = None
        self.vec_norm = None
        self.algo_type = None
        
        # Determine action levels based on model type
        # PPO models typically use 6 action levels: (0, 10, 20, 40, 60, 80)
        # DQN models might use different action levels
        if "ppo_best_98_service" in os.path.basename(model_path):
            self.action_levels = [0, 10, 20, 40, 60, 80]  # 6 levels for PPO
            print(f"Using PPO action levels: {self.action_levels}")
        else:
            self.action_levels = [0, 10, 20, 40, 60, 80]  # Default to 6 levels
            print(f"Using default action levels: {self.action_levels}")
        
        # Try to load model if it exists
        if os.path.exists(model_path) and SB3_AVAILABLE:
            try:
                # Determine model type from path
                model_name = os.path.basename(model_path).lower()
                if "dqn" in model_name:
                    self.model = DQN.load(model_path)
                    self.algo_type = "dqn"
                elif "ppo" in model_name:
                    self.model = PPO.load(model_path)
                    self.algo_type = "ppo"
                else:
                    # Try PPO first for newer models
                    try:
                        self.model = PPO.load(model_path)
                        self.algo_type = "ppo"
                    except:
                        self.model = DQN.load(model_path)
                        self.algo_type = "dqn"
                        
                print(f"✅ Loaded {self.algo_type.upper()} model from {model_path}")
                
                # Try to load VecNormalize stats
                self._load_vec_normalize()
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.model = None
        else:
            if not os.path.exists(model_path):
                print(f"❌ Model not found at {model_path}")
            if not SB3_AVAILABLE:
                print("❌ stable-baselines3 not installed")
    
    def _load_vec_normalize(self):
        """Load VecNormalize statistics if available."""
        # Try multiple naming conventions for VecNormalize files
        model_base = os.path.basename(self.model_path)[:-4]  # Remove .zip
        
        possible_paths = [
            os.path.join(os.path.dirname(self.model_path), f"vecnorm_{model_base}.pkl"),
            os.path.join(os.path.dirname(self.model_path), f"vecnorm_{self.algo_type}_W{self.n_warehouses}.pkl"),
            os.path.join(os.path.dirname(self.model_path), f"vecnorm_{self.algo_type}.pkl")
        ]
        
        for vecnorm_path in possible_paths:
            if os.path.exists(vecnorm_path):
                try:
                    # Create dummy env to load normalization
                    from env.inventory_env import InventoryEnv
                    
                    def make_env():
                        env = InventoryEnv(
                            n_warehouses=self.n_warehouses,
                            horizon=52,
                            action_levels=tuple(self.action_levels)
                        )
                        if self.algo_type == "dqn":
                            from env.wrappers import MultiDiscreteToDiscreteWrapper
                            env = MultiDiscreteToDiscreteWrapper(env)
                        return env
                    
                    vec_env = DummyVecEnv([make_env])
                    self.vec_norm = VecNormalize.load(vecnorm_path, vec_env)
                    self.vec_norm.training = False
                    self.vec_norm.norm_reward = False
                    print(f"✅ Loaded VecNormalize from {vecnorm_path}")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load VecNormalize from {vecnorm_path}: {e}")
                    continue
        
        print("⚠️ No VecNormalize file found - using raw observations")
    
    def predict(self, obs):
        """
        Predict action given observation.
        
        Args:
            obs: Current observation
            
        Returns:
            Action indices for each warehouse
        """
        if self.model is not None:
            try:
                # Normalize observation if vec_norm is available
                if self.vec_norm is not None:
                    # Ensure obs is 2D for VecNormalize
                    if obs.ndim == 1:
                        obs_norm = self.vec_norm.normalize_obs(obs.reshape(1, -1))
                        obs_to_use = obs_norm.flatten()
                    else:
                        obs_to_use = self.vec_norm.normalize_obs(obs)
                else:
                    obs_to_use = obs
                
                # Ensure correct shape for model prediction
                if obs_to_use.ndim == 1:
                    obs_to_use = obs_to_use.reshape(1, -1)
                
                # Get action from model
                action, _states = self.model.predict(obs_to_use, deterministic=True)
                
                if self.algo_type == "dqn":
                    # DQN returns a flat integer, convert to multi-discrete indices
                    flat_action = int(action[0]) if hasattr(action, '__len__') else int(action)
                    n_actions = len(self.action_levels)  # Use actual action levels length
                    indices = []
                    a = flat_action
                    for _ in range(self.n_warehouses):
                        indices.append(a % n_actions)
                        a //= n_actions
                    return np.array(indices, dtype=np.int64)
                else:
                    # PPO returns multi-discrete indices directly
                    if hasattr(action, 'flatten'):
                        action_indices = action.flatten().astype(np.int64)
                    else:
                        action_indices = np.array(action, dtype=np.int64).flatten()
                    
                    # Ensure we have the right number of actions
                    if len(action_indices) < self.n_warehouses:
                        action_indices = np.pad(action_indices, (0, self.n_warehouses - len(action_indices)), 'constant')
                    elif len(action_indices) > self.n_warehouses:
                        action_indices = action_indices[:self.n_warehouses]
                    
                    # Clamp actions to valid range
                    action_indices = np.clip(action_indices, 0, len(self.action_levels) - 1)
                    
                    return action_indices
                    
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                print(f"   Observation shape: {obs.shape}")
                print(f"   Observation: {obs[:min(8, len(obs))]}")
                return self._fallback_policy(obs)
        else:
            # No model loaded, use fallback
            return self._fallback_policy(obs)
    
    def get_action(self, obs):
        """Alias for predict."""
        return self.predict(obs)
    
    def _fallback_policy(self, obs):
        """Simple fallback policy (order-up-to)."""
        try:
            # Use order-up-to policy as fallback
            current_stock = obs[:self.n_warehouses]
            target = 80  # Target inventory level
            
            actions = []
            for stock in current_stock:
                gap = max(0, target - stock)
                # Find closest action level index
                action_idx = np.argmin([abs(a - gap) for a in self.action_levels])
                actions.append(action_idx)
            
            return np.array(actions, dtype=np.int64)
        except Exception as e:
            print(f"⚠️ Fallback policy failed: {e}")
            # Emergency fallback: no order
            return np.zeros(self.n_warehouses, dtype=np.int64)
    
    def predict_with_fallback(self, obs, stockout_threshold: float = 20):
        """
        Predict with safety fallback.
        
        Args:
            obs: Current observation
            stockout_threshold: Stock level below which to use fallback
            
        Returns:
            Action indices
        """
        current_stock = obs[:self.n_warehouses]
        
        # Use fallback if stock is critically low
        if self.use_fallback and np.any(current_stock < stockout_threshold):
            return self._fallback_policy(obs)
        else:
            return self.predict(obs)