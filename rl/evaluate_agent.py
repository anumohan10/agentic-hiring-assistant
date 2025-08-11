#rl>evaluate_agent.py
import argparse, json, os
from pathlib import Path
import numpy as np
from gymnasium import spaces

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.inventory_env import InventoryEnv
from env.wrappers import MultiDiscreteToDiscreteWrapper


def load_model(path):
    name = os.path.basename(path).lower()
    if "dqn" in name:
        return DQN.load(path), "dqn"
    if "ppo" in name:
        return PPO.load(path), "ppo"
    try:
        return DQN.load(path), "dqn"
    except Exception:
        return PPO.load(path), "ppo"


def make_vec_env(env_kwargs, wrap_for_dqn: bool):
    def _f():
        env = InventoryEnv(**env_kwargs)
        if wrap_for_dqn:
            env = MultiDiscreteToDiscreteWrapper(env)
        return env
    return DummyVecEnv([_f])


def load_vecnorm(vec_env, stats_path):
    """Load VecNormalize stats onto an existing vec env for eval."""
    vecnorm = VecNormalize.load(stats_path, vec_env)
    vecnorm.training = False      # eval mode
    vecnorm.norm_reward = False   # keep rewards in env scale for reporting
    return vecnorm


def run_eval_vec(model, episodes, vec_env, per_step_log_path=None):
    step_logs, ep_metrics = [], []

    # unwrap for logging internals
    base_env = vec_env.venv.envs[0].unwrapped
    inner_env = vec_env.venv.envs[0]
    is_dqn_wrapper = isinstance(inner_env, MultiDiscreteToDiscreteWrapper)

    for ep in range(episodes):
        # Seed the underlying env(s), then reset the VecNormalize wrapper
        vec_env.venv.env_method("reset", seed=ep)   # <- seed inner env
        obs = vec_env.reset()                       # <- reset VecNormalize (no seed arg)

        done_flag = False
        total_cost = holding_sum = stockout_sum = order_sum = 0.0
        svc_levels = []

        while not done_flag:
            action, _ = model.predict(obs, deterministic=True)

            logged_action = action[0]
            if is_dqn_wrapper:
                try:
                    logged_action = inner_env.action(int(action[0]))
                except Exception:
                    pass

            obs, rewards, dones, infos = vec_env.step(action)
            done_flag = bool(dones[0])

            info = infos[0]
            cost = float(infos[0]["holding_cost"] + infos[0]["stockout_cost"] + infos[0]["order_cost"] + infos[0].get("transfer_cost", 0.0))
            total_cost += cost
            holding_sum += info["holding_cost"]
            stockout_sum += info["stockout_cost"]
            order_sum += info["order_cost"]
            svc_levels.append(info["service_level"])

            if per_step_log_path:
                step_logs.append({
                    "episode": ep,
                    "t": base_env.t,
                    "action": (logged_action.tolist() if hasattr(logged_action, "tolist") else logged_action),
                    "cost": cost,
                    "holding_cost": info["holding_cost"],
                    "stockout_cost": info["stockout_cost"],
                    "order_cost": info["order_cost"],
                    "service_level": info["service_level"],
                    "stockout_mult": info.get("stockout_mult"),
                    "demand": (info["demand"].tolist() if hasattr(info["demand"], "tolist") else info["demand"]),
                    "fulfilled": (info["fulfilled"].tolist() if hasattr(info["fulfilled"], "tolist") else info["fulfilled"]),
                })

        ep_metrics.append({
            "total_cost": total_cost,
            "avg_service": float(np.mean(svc_levels)),
            "holding_cost": holding_sum,
            "stockout_cost": stockout_sum,
            "order_cost": order_sum,
        })


    if per_step_log_path:
        Path(os.path.dirname(per_step_log_path)).mkdir(parents=True, exist_ok=True)
        with open(per_step_log_path, "w") as f:
            for row in step_logs:
                f.write(json.dumps(row) + "\n")

    summary = {
        "episodes": episodes,
        "mean_total_cost": float(np.mean([m["total_cost"] for m in ep_metrics])),
        "std_total_cost": float(np.std([m["total_cost"] for m in ep_metrics])),
        "mean_service": float(np.mean([m["avg_service"] for m in ep_metrics])),
        "mean_holding": float(np.mean([m["holding_cost"] for m in ep_metrics])),
        "mean_stockout": float(np.mean([m["stockout_cost"] for m in ep_metrics])),
        "mean_order": float(np.mean([m["order_cost"] for m in ep_metrics])),
    }
    return summary, ep_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--warehouses", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--target_service", type=float, default=0.90)
    ap.add_argument("--below_target_mult", type=float, default=1.5)
    ap.add_argument("--out", default=None, help="Summary JSON path (default auto in results/eval)")
    ap.add_argument("--per_step_log", default=None, help="Write JSONL of per-step rows")
    args = ap.parse_args()

    model, algo = load_model(args.model_path)
    env_kwargs = dict(
    n_warehouses=args.warehouses,
    horizon=args.horizon,
    target_service=args.target_service,         # ← new
    below_target_mult=args.below_target_mult,   # ← new
)


    # Build vec env & load normalization stats saved at training (if present)
    wrap_for_dqn = (algo == "dqn")
    raw_vec_env = make_vec_env(env_kwargs, wrap_for_dqn)
    stats_path = os.path.join("results", "models", f"vecnorm_{algo}_W{args.warehouses}.pkl")
    if os.path.exists(stats_path):
        vec_env = load_vecnorm(raw_vec_env, stats_path)
    else:
        # Fallback: create a non-training VecNormalize so obs are normalized similarly
        vec_env = VecNormalize(raw_vec_env, training=False, norm_obs=True, norm_reward=False)

    # outputs
    if args.out is None:
        out_dir = "results/eval"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        args.out = os.path.join(out_dir, f"{algo}_eval.json")
    if args.per_step_log is None:
        args.per_step_log = os.path.join("results/eval", f"{algo}_rollouts.jsonl")

    # Vectorized evaluation (uses same normalization pipeline)
    summary, ep_metrics = run_eval_vec(model, args.episodes, vec_env, args.per_step_log)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved per-step logs → {args.per_step_log}")
    print(f"Saved summary → {args.out}")


if __name__ == "__main__":
    main()
