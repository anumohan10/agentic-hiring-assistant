#rl> train_rl_agent.py
import argparse, os, json, random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed


from env.inventory_env import InventoryEnv
from env.wrappers import MultiDiscreteToDiscreteWrapper

ALGOS = {"dqn": DQN, "ppo": PPO}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


def make_env(env_kwargs, use_dqn_wrapper: bool):
    def _f():
        env = InventoryEnv(**env_kwargs)
        if use_dqn_wrapper:
            env = MultiDiscreteToDiscreteWrapper(env)  # wrap only for DQN
        return Monitor(env)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warehouses", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--target_service", type=float, default=0.90)
    ap.add_argument("--below_target_mult", type=float, default=1.5)
    ap.add_argument("--models_dir", default="results/models")
    ap.add_argument("--tb_dir", default="results/tb")
    ap.add_argument("--checkpoint_every", type=int, default=0, help="Save model every N episodes (0=off)")
    args = ap.parse_args()

    seed_everything(args.seed)

    # ---- Env ----
    env_kwargs = dict(
    n_warehouses=args.warehouses,
    horizon=args.horizon,
    target_service=args.target_service,
    below_target_mult=args.below_target_mult,
)
    use_wrapper = args.algo == "dqn"
    vec_env = DummyVecEnv([make_env(env_kwargs, use_wrapper)])
    # ⬇️ normalize observations + rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # ---- Model ----
    if args.algo == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.tb_dir,
            learning_rate=2.5e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=1,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[128, 128]),
            seed=args.seed,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.tb_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128]),
            seed=args.seed,
        )

    total_timesteps = args.episodes * args.horizon
    print(f"Training {args.algo.upper()} for {total_timesteps:,} timesteps "
          f"({args.episodes} eps × {args.horizon} steps)")


    # ---- Train with optional checkpoints ----
    ckpt_every_steps = args.checkpoint_every * args.horizon if args.checkpoint_every > 0 else None
    steps_done = 0
    while True:
        to_do = total_timesteps - steps_done
        if to_do <= 0:
            break
        chunk = min(to_do, ckpt_every_steps or to_do)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=True)
        steps_done += chunk

        if ckpt_every_steps:
            Path(args.models_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(args.models_dir, f"{args.algo}_W{args.warehouses}_ckpt_{steps_done}.zip")
            model.save(ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")

    # ---- Save final ----
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.models_dir, f"{args.algo}_W{args.warehouses}.zip")
    model.save(out_path)
    # ⬇️ save vecnorm stats alongside the model
    vecnorm_path = os.path.join(args.models_dir, f"vecnorm_{args.algo}_W{args.warehouses}.pkl")
    vec_env.save(vecnorm_path)
    print(f"🧮 Saved VecNormalize stats → {vecnorm_path}")
    meta = {
        "algo": args.algo,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "model_path": out_path,
        "target_service": args.target_service,
        "below_target_mult": args.below_target_mult,
    }
    with open(os.path.join(args.models_dir, f"{args.algo}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved final model → {out_path}")
    print(f"💡 TensorBoard: tensorboard --logdir {args.tb_dir}")


if __name__ == "__main__":
    main()
