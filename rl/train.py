"""
Train DQN, PPO, and A2C on ElementQuestEnv.
Logs metrics to training_logs/<algo>_log.csv for live dashboard.

Usage:
    python rl/train.py --algo dqn
    python rl/train.py --algo ppo
    python rl/train.py --algo a2c
    python rl/train.py --algo all          # trains all three sequentially
    python rl/train.py --algo dqn --render-every 100   # live preview every 100 episodes
    python rl/train.py --reset             # wipe all logs and saved models, then exit
"""
import argparse
import csv
import shutil
import time
from pathlib import Path

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from rl.env import ElementQuestEnv

MODELS_DIR = Path("rl/models")
LOGS_DIR   = Path("training_logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Hyperparameters
# ------------------------------------------------------------------ #
ALGO_CONFIGS = {
    "dqn": dict(
        cls=DQN,
        total_timesteps=300_000,
        kwargs=dict(
            policy="MlpPolicy",
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            train_freq=4,
            target_update_interval=500,
            verbose=0,
        )
    ),
    "ppo": dict(
        cls=PPO,
        total_timesteps=500_000,
        kwargs=dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
        )
    ),
    "a2c": dict(
        cls=A2C,
        total_timesteps=400_000,
        kwargs=dict(
            policy="MlpPolicy",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            verbose=0,
        )
    ),
}


# ------------------------------------------------------------------ #
# Logging callback
# ------------------------------------------------------------------ #
class TrainingLogger(BaseCallback):
    """Writes per-episode stats to CSV for the live dashboard."""

    def __init__(self, algo_name: str, log_path: Path):
        super().__init__()
        self.algo_name   = algo_name
        self.log_path    = log_path
        self._ep_reward  = 0.0
        self._episode    = 0
        self._start_time = time.time()

        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'episode', 'timestep', 'reward', 'ep_length', 'elapsed_s', 'algo'
            ])

    def _on_step(self) -> bool:
        self._ep_reward += self.locals.get('rewards', [0])[0]
        done = self.locals.get('dones', [False])[0]

        if done:
            self._episode += 1
            ep_len  = self.locals.get('infos', [{}])[0].get('episode', {}).get('l', 0)
            elapsed = time.time() - self._start_time

            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self._episode, self.num_timesteps,
                    round(self._ep_reward, 3), ep_len,
                    round(elapsed, 1), self.algo_name,
                ])

            if self._episode % 50 == 0:
                print(f"[{self.algo_name.upper()}] ep {self._episode:4d} | "
                      f"steps {self.num_timesteps:7d} | "
                      f"reward {self._ep_reward:7.1f}")
            self._ep_reward = 0.0
        return True


# ------------------------------------------------------------------ #
# Live render callback
# ------------------------------------------------------------------ #
class LiveRenderCallback(BaseCallback):
    """
    Every `render_every` training episodes, runs one full episode
    in a rendered Pygame window so you can watch the agent's current behaviour.
    Close the preview window or press ESC to skip and keep training.
    """

    def __init__(self, render_every: int = 100):
        super().__init__()
        self.render_every = render_every
        self._episode     = 0
        self._render_env  = None

    def _on_step(self) -> bool:
        done = self.locals.get('dones', [False])[0]
        if done:
            self._episode += 1
            if self._episode % self.render_every == 0:
                self._run_preview()
        return True

    def _run_preview(self):
        import pygame
        print(f"  ▶ Live preview — episode {self._episode} "
              f"(close window or ESC to continue training)")

        # Lazy-init the render env
        if self._render_env is None:
            self._render_env = ElementQuestEnv(render_mode='human')

        # Update window title so it's clear this is a preview
        pygame.display.set_caption(
            f"Element Quest — Training Preview  [ep {self._episode}]"
        )

        obs, _ = self._render_env.reset()
        done   = False
        total_reward = 0.0
        skipped      = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._render_env.close()
                    self._render_env = None
                    skipped = True
                    done    = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    skipped = True
                    done    = True
                    break

            if done:
                break

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self._render_env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        if not skipped:
            print(f"  ✓ Preview done — reward: {total_reward:.1f}")

    def _on_training_end(self):
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None


# ------------------------------------------------------------------ #
# Reset helper
# ------------------------------------------------------------------ #
def reset_all():
    """Delete all training logs and saved models."""
    deleted = []
    for f in LOGS_DIR.glob("*.csv"):
        f.unlink()
        deleted.append(str(f))
    for f in MODELS_DIR.glob("*.zip"):
        f.unlink()
        deleted.append(str(f))
    if deleted:
        print("Deleted:")
        for d in deleted:
            print(f"  {d}")
    else:
        print("Nothing to delete — already clean.")


# ------------------------------------------------------------------ #
# Train one algorithm
# ------------------------------------------------------------------ #
def train_algo(algo_name: str, render_every: int = 0):
    print(f"\n{'='*50}")
    print(f"  Training {algo_name.upper()}")
    if render_every > 0:
        print(f"  Live preview every {render_every} episodes")
    print(f"{'='*50}")

    cfg        = ALGO_CONFIGS[algo_name]
    log_path   = LOGS_DIR   / f"{algo_name}_log.csv"
    model_path = MODELS_DIR / f"{algo_name}_model"

    env   = Monitor(ElementQuestEnv())
    model = cfg['cls'](env=env, **cfg['kwargs'])

    callbacks = [TrainingLogger(algo_name, log_path)]
    if render_every > 0:
        callbacks.append(LiveRenderCallback(render_every=render_every))

    model.learn(
        total_timesteps=cfg['total_timesteps'],
        callback=CallbackList(callbacks),
    )

    model.save(str(model_path))
    print(f"\n  Saved → {model_path}.zip")
    env.close()


# ------------------------------------------------------------------ #
# Load a saved model
# ------------------------------------------------------------------ #
def load_model(algo_name: str):
    cfg        = ALGO_CONFIGS[algo_name]
    model_path = MODELS_DIR / f"{algo_name}_model"
    if not model_path.with_suffix('.zip').exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}.zip — train first.")
    return cfg['cls'].load(str(model_path))


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='dqn',
                        choices=['dqn', 'ppo', 'a2c', 'all'])
    parser.add_argument('--render-every', type=int, default=0,
                        help='Show a live preview window every N training episodes. '
                             '0 = disabled (default).')
    parser.add_argument('--reset', action='store_true',
                        help='Delete all training logs and saved models, then exit.')
    args = parser.parse_args()

    if args.reset:
        reset_all()
        return

    render_every = args.render_every

    if args.algo == 'all':
        for name in ['dqn', 'ppo', 'a2c']:
            train_algo(name, render_every=render_every)
    else:
        train_algo(args.algo, render_every=render_every)


if __name__ == '__main__':
    main()
