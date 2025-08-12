import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import sys
sys.path.append('.')

from env.inventory_env import InventoryEnv
from env.wrappers import MultiDiscreteToDiscreteWrapper

def test_model(model_path, vecnorm_path):
    """Test if a model loads and produces different actions."""
    print(f"\nTesting: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model file not found!")
        return
    
    # Load model
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
        algo = "dqn"
    else:
        model = PPO.load(model_path)
        algo = "ppo"
    
    print(f"  ✅ Model loaded ({algo})")
    
    # Create test environment
    env = InventoryEnv(
        n_warehouses=2,
        horizon=52,
        action_levels=(0, 10, 20, 40, 60, 80)
    )
    
    if algo == "dqn":
        env = MultiDiscreteToDiscreteWrapper(env)
    
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if exists
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  ✅ VecNormalize loaded")
    else:
        print(f"  ⚠️ No VecNormalize found at {vecnorm_path}")
    
    # Test predictions on different observations
    obs = vec_env.reset()
    actions_taken = []
    
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        # Handle both single values and arrays
        if isinstance(action[0], np.ndarray):
            actions_taken.append(tuple(action[0]))  # Convert array to tuple for hashing
        else:
            actions_taken.append(int(action[0]))
        obs, _, _, _ = vec_env.step(action)
    
    print(f"  Actions taken: {actions_taken}")
    print(f"  Unique actions: {len(set(actions_taken))} different")
    
    return actions_taken

# Test all models
models = [
    ("results/models/dqn_improved.zip", "results/models/vecnorm_dqn_improved.pkl"),
    ("results/models/ppo_balanced_92_gain8.zip", "results/models/vecnorm_ppo_balanced_92_gain8.pkl"),
    ("results/models/ppo_best.zip", "results/models/vecnorm_ppo_best.pkl"),
    ("results/models/ppo_high_service_95_gain10.zip", "results/models/vecnorm_ppo_high_service_95_gain10.pkl")
]

all_actions = {}
for model_path, vecnorm_path in models:
    actions = test_model(model_path, vecnorm_path)
    if actions:
        all_actions[model_path] = actions

# Check if PPO models are different
ppo_models = [k for k in all_actions.keys() if "ppo" in k]
if len(ppo_models) > 1:
    print("\n" + "="*50)
    print("Checking if PPO models produce different actions:")
    for i, model1 in enumerate(ppo_models):
        for model2 in ppo_models[i+1:]:
            if all_actions[model1] == all_actions[model2]:
                print(f"⚠️ {os.path.basename(model1)} and {os.path.basename(model2)} produce IDENTICAL actions!")
            else:
                print(f"✅ {os.path.basename(model1)} and {os.path.basename(model2)} are different")