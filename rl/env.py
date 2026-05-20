"""
Custom Gymnasium environment wrapping GameEngine.
Single-agent: FireAgent only.
Observation: feature vector (no pixels → CPU trainable).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine
from game.tiles import Tile, TILE_SIZE, GRID_COLS, GRID_ROWS
from game.characters import ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP

# Reward constants
R_GEM      = 10.0
R_WIN      = 150.0
R_DEATH    = -100.0
R_TIME     = -0.05   # per RL step
R_PROGRESS = 0.3     # shaping: moving toward nearest gem

MAX_STEPS  = 1200    # episode cap


class ElementQuestEnv(gym.Env):
    """
    Observation space  : Box(float32, shape=(35,))
    Action space       : Discrete(4)  {noop, left, right, jump}

    Parameters
    ----------
    render_fps : int
        Max FPS when render_mode='human'. 0 = uncapped.
        Use a high value (e.g. 300) during training so the display
        doesn't bottleneck the training loop.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, map_module=None, render_mode=None,
                 frame_skip: int = 4, render_fps: int = 60):
        super().__init__()
        if map_module is None:
            from game.maps import map_race
            map_module = map_race

        self.map_module   = map_module
        self.render_mode  = render_mode
        self.frame_skip   = frame_skip
        self.render_fps   = render_fps
        self._step_count  = 0
        self._prev_gem_dist = None

        self.engine = GameEngine(map_module, agents='fire', render_mode=render_mode)

        OBS_SIZE = 35
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self._step_count    = 0
        self._prev_gem_dist = None
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        reward     = 0.0
        terminated = False
        truncated  = False

        for _ in range(self.frame_skip):
            events = self.engine.step(fire_action=int(action))

            if self.render_mode == 'human':
                self.engine.render()
                self._handle_window_events()
                if self.render_fps > 0:
                    self.engine.tick(self.render_fps)

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
        reward += R_TIME

        # Shaping: reward moving closer to nearest gem
        gem_positions = self.engine.gem_positions(int(Tile.FIRE_GEM))
        if gem_positions and not terminated:
            fc  = self.engine.fire_char
            cx  = fc.px + 16
            cy  = fc.py + 20
            dists   = [abs(cx - g[0]*TILE_SIZE) + abs(cy - g[1]*TILE_SIZE)
                       for g in gem_positions]
            nearest = min(dists)
            if self._prev_gem_dist is not None and nearest < self._prev_gem_dist:
                reward += R_PROGRESS * ((self._prev_gem_dist - nearest) / TILE_SIZE)
            self._prev_gem_dist = nearest

        if self._step_count >= MAX_STEPS:
            truncated = True

        info = {
            'gems':  self.engine.fire_char.gems,
            'alive': self.engine.fire_char.alive,
            'step':  self._step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _handle_window_events(self):
        """
        Pump pygame events to keep the window responsive.
        If the user closes the window, silently switch to headless mode
        so training continues without crashing.
        """
        if self.render_mode != 'human':
            return
        try:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Switch to headless — training keeps running
                    self.render_mode        = None
                    self.engine.render_mode = None
                    pygame.quit()
                    print("\n[Render] Window closed — continuing headless.")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == 'human':
            self.engine.render()

    def close(self):
        self.engine.close()

    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        fc   = self.engine.fire_char
        grid = self.engine.grid

        norm_x  = fc.px / (GRID_COLS * TILE_SIZE)
        norm_y  = fc.py / (GRID_ROWS * TILE_SIZE)
        norm_vx = np.clip(fc.vx / 10.0, -1, 1)
        norm_vy = np.clip(fc.vy / 16.0, -1, 1)
        on_gnd  = float(fc.on_ground)

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

        door_pos = self.engine.gem_positions(int(Tile.FIRE_DOOR))
        if door_pos:
            d = door_pos[0]
            door_dx = np.clip((d[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
            door_dy = np.clip((d[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
        else:
            door_dx, door_dy = 0.0, 0.0

        gems_remaining = len(gem_pos) / max(self.engine.total_gems, 1)

        col_c = int(cx // TILE_SIZE)
        row_c = int(cy // TILE_SIZE)
        local = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row_c + dr, col_c + dc
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    local.append(grid[r][c] / 9.0)
                else:
                    local.append(1.0)  # treat out-of-bounds as wall

        return np.array([
            norm_x, norm_y, norm_vx, norm_vy, on_gnd,
            gem_dx, gem_dy, door_dx, door_dy,
            gems_remaining,
            *local,
        ], dtype=np.float32)
