# test_orchestrator_simple.py
"""
Simple test to verify orchestrator works with your exact setup.
Run from project root: python test_orchestrator_simple.py
"""
import numpy as np
import sys
import os
sys.path.append('.')

from env.inventory_env import InventoryEnv
from agents.orchestrator import Orchestrator
from agents.restock_recommender_agent import order_up_to_S, fixed_order_policy
from custom_tools.cost_simulation_tool import CostParams

def test_orchestrator():
    print("="*60)
    print("TESTING ORCHESTRATOR")
    print("="*60)
    
    # 1. Initialize environment with your exact parameters
    print("\n1. Initializing environment...")
    env = InventoryEnv(
        n_warehouses=2,
        horizon=52,
        action_levels=(0, 10, 20, 40, 60, 80),  # Your exact action levels
        cost_params=CostParams(holding_cost=1.0, stockout_cost=10.0, order_cost=2.0),
        target_service=0.92,
        below_target_mult=8.0
    )
    print(f"   ✅ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Action levels: {env.action_levels}")
    
    # 2. Initialize orchestrator
    print("\n2. Initializing orchestrator...")
    orchestrator = Orchestrator(
        action_levels=(0, 10, 20, 40, 60, 80),
        n_warehouses=2,
        target_service=0.92,
        below_target_mult=8.0
    )
    print("   ✅ Orchestrator created")
    
    # 3. Add heuristic policies
    print("\n3. Adding policies...")
    
    # Add order-up-to-S policy
    orchestrator.add_policy(
        "order_up_to_s",
        lambda obs: order_up_to_S(
            obs, 
            S=80, 
            action_levels=(0, 10, 20, 40, 60, 80), 
            n_warehouses=2
        )
    )
    print("   ✅ Added order-up-to-S policy")
    
    # Add fixed order policy
    orchestrator.add_policy(
        "fixed_order",
        lambda obs: fixed_order_policy(
            obs, 
            action_levels=(0, 10, 20, 40, 60, 80), 
            fixed=20, 
            n_warehouses=2
        )
    )
    print("   ✅ Added fixed order policy")
    
    # Try to add RL policies if they exist
    model_paths = {
        "ppo": ["results/models/ppo_best.zip", "results/models/ppo_W2.zip"],
        "dqn": ["results/models/dqn_improved.zip", "results/models/dqn_W2.zip"]
    }
    
    for algo, paths in model_paths.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    orchestrator.add_rl_policy(algo, path, algo)
                    print(f"   ✅ Added {algo.upper()} policy from {path}")
                    break
                except Exception as e:
                    print(f"   ⚠️ Could not add {algo.upper()}: {e}")
    
    print(f"   Total policies: {len(orchestrator.policies)}")
    
    # 4. Initialize bandit
    orchestrator.initialize_bandit()
    print("\n4. ✅ Initialized UCB bandit")
    
    # 5. Run a short test episode
    print("\n5. Running test episode...")
    print("-"*40)
    
    obs, _ = env.reset(seed=42)
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial stock levels: {obs[:2]}")
    
    total_reward = 0
    for step in range(10):  # Just 10 steps for quick test
        # Get action from orchestrator
        action_indices = orchestrator.step(obs)
        
        # Verify action indices are valid
        assert len(action_indices) == 2, f"Expected 2 actions, got {len(action_indices)}"
        assert all(0 <= idx < 6 for idx in action_indices), f"Invalid action indices: {action_indices}"
        
        # Convert to values for display
        action_values = env.action_levels[action_indices]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_indices)
        total_reward += reward
        
        # Update orchestrator metrics
        orchestrator.update_step_metrics(info)
        
        # Display first few steps
        if step < 3:
            policy = orchestrator.episode_meta[-1]["policy"] if orchestrator.episode_meta else "unknown"
            print(f"   Step {step}: Policy={policy}, Actions={action_values}, "
                  f"Service={info['service_level']:.2%}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    # End episode
    orchestrator.end_episode()
    
    # 6. Show summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    summary = orchestrator.get_performance_summary()
    print(f"Episodes completed: {summary.get('episodes_run', 0)}")
    
    if summary.get('episodes_run', 0) > 0:
        print(f"Mean cost: ${summary['mean_cost']:.2f}")
        print(f"Mean service: {summary['mean_service']:.2%}")
    
    print(f"Policies available: {summary.get('policies_used', [])}")
    
    if summary.get('bandit_stats'):
        stats = summary['bandit_stats']
        print(f"Bandit pulls: {stats.get('total_pulls', 0)}")
        print(f"Policy counts: {stats.get('counts', {})}")
    
    print("\n✅ Orchestrator test completed successfully!")
    print("   The orchestrator is ready to use in your project.")
    
    return orchestrator

if __name__ == "__main__":
    try:
        orchestrator = test_orchestrator()
        print("\n🎉 ALL TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()