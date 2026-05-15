"""
Custom Gymnasium environment wrapping GameEngine.
Single-agent: FireAgent only.
Observation: feature vector (no pixels → CPU trainable).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.engine import GameEngine
from game.tiles import Tile, TILE_SIZE, GRID_COLS, GRID_ROWS
from game.characters import ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP

# Reward constants
R_GEM       = 10.0
R_WIN       = 150.0
R_DEATH     = -100.0
R_TIME      = -0.05   # per step
R_PROGRESS  = 0.3     # shaping: moving toward nearest gem

MAX_STEPS   = 1200    # episode cap


class ElementQuestEnv(gym.Env):
    """
    Observation space  : Box(float32, shape=(obs_size,))
    Action space       : Discrete(4)  {noop, left, right, jump}

    State features:
      - fire agent (x, y) normalised                         2
      - fire agent velocity (vx, vy) normalised              2
      - on_ground                                            1
      - relative (dx, dy) to nearest fire gem (normalised)  2
      - relative (dx, dy) to fire door (normalised)         2
      - gems_remaining normalised                            1
      - local 5x5 tile grid around agent (flattened)        25
      Total                                                 35
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, map_module=None, render_mode=None, frame_skip: int = 4):
        super().__init__()
        if map_module is None:
            from game.maps import map_training
            map_module = map_training

        self.map_module   = map_module
        self.render_mode  = render_mode
        self.frame_skip   = frame_skip
        self._step_count  = 0

        self.engine = GameEngine(map_module, agents='fire', render_mode=render_mode)

        OBS_SIZE = 35
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # noop, left, right, jump

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self._step_count = 0
        self._prev_gem_dist = None
        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated  = False

        prev_gems = self.engine.fire_char.gems

        for _ in range(self.frame_skip):
            events = self.engine.step(fire_action=int(action))
            if self.render_mode == 'human':
                self.engine.render()
                self.engine.tick(60)

            fire_ev = events.get('fire', {})

            if 'collect_gem' in fire_ev:
                reward += R_GEM
            if 'reached_door' in fire_ev:
                reward += R_WIN
                terminated = True
                break
            if 'died' in fire_ev:
                reward += R_DEATH
                terminated = True
                break

        self._step_count += 1
        reward += R_TIME  # time penalty per RL step

        # Shaping: reward for moving closer to nearest gem
        gem_positions = self.engine.gem_positions(int(Tile.FIRE_GEM))
        if gem_positions and not terminated:
            fc = self.engine.fire_char
            cx = fc.px + 16
            cy = fc.py + 20
            dists = [abs(cx - g[0] * TILE_SIZE) + abs(cy - g[1] * TILE_SIZE)
                     for g in gem_positions]
            nearest = min(dists)
            if self._prev_gem_dist is not None:
                delta = self._prev_gem_dist - nearest
                if delta > 0:
                    reward += R_PROGRESS * (delta / TILE_SIZE)
            self._prev_gem_dist = nearest

        if self._step_count >= MAX_STEPS:
            truncated = True

        obs = self._get_obs()
        info = {
            'gems': self.engine.fire_char.gems,
            'alive': self.engine.fire_char.alive,
            'step': self._step_count,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == 'human':
            self.engine.render()

    def close(self):
        self.engine.close()

    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        fc = self.engine.fire_char
        grid = self.engine.grid

        norm_x  = fc.px / (GRID_COLS * TILE_SIZE)
        norm_y  = fc.py / (GRID_ROWS * TILE_SIZE)
        norm_vx = np.clip(fc.vx / 10.0, -1, 1)
        norm_vy = np.clip(fc.vy / 16.0, -1, 1)
        on_gnd  = float(fc.on_ground)

        # Nearest fire gem
        gem_pos = self.engine.gem_positions(int(Tile.FIRE_GEM))
        cx = fc.px + 16
        cy = fc.py + 20
        if gem_pos:
            nearest = min(gem_pos,
                          key=lambda g: abs(cx - g[0]*TILE_SIZE) + abs(cy - g[1]*TILE_SIZE))
            gem_dx = np.clip((nearest[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
            gem_dy = np.clip((nearest[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
        else:
            gem_dx, gem_dy = 0.0, 0.0

        # Door position
        door_pos = self.engine.gem_positions(int(Tile.FIRE_DOOR))
        if door_pos:
            d = door_pos[0]
            door_dx = np.clip((d[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
            door_dy = np.clip((d[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
        else:
            door_dx, door_dy = 0.0, 0.0

        gems_remaining = len(gem_pos) / max(self.engine.total_gems, 1)

        # Local 5x5 tile window
        col_c = int(cx // TILE_SIZE)
        row_c = int(cy // TILE_SIZE)
        local = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row_c + dr, col_c + dc
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    local.append(grid[r][c] / 9.0)
                else:
                    local.append(1.0)  # wall

        obs = np.array([
            norm_x, norm_y, norm_vx, norm_vy, on_gnd,
            gem_dx, gem_dy, door_dx, door_dy,
            gems_remaining,
            *local,
        ], dtype=np.float32)
        return obs
