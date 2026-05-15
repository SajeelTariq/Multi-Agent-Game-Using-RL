"""Core game engine — drives physics, rendering, and game state."""
from __future__ import annotations
import copy
import pygame
from game.tiles import (
    Tile, TILE_SIZE, GRID_COLS, GRID_ROWS,
    UI_HEIGHT, WINDOW_W, WINDOW_H, BG_COLOR, draw_tile,
)
from game.characters import Character, CharacterType


class GameEngine:
    """
    Manages game state for one session (one map).

    Parameters
    ----------
    map_module : module with a load() function returning (grid, fire_spawn, water_spawn)
    agents     : 'fire' | 'both'  — which characters are active
    render_mode: 'human' | None
    """

    def __init__(self, map_module, agents: str = 'fire', render_mode=None):
        self.map_module = map_module
        self.agents = agents
        self.render_mode = render_mode

        self._base_grid, self.fire_spawn, self.water_spawn = map_module.load()
        self.grid: list[list[int]] = []

        self.fire_char: Character | None = None
        self.water_char: Character | None = None

        self.frame = 0
        self.total_gems = 0

        if render_mode == 'human':
            self._init_pygame()

        self.reset()

    # ------------------------------------------------------------------ #
    # Pygame init
    # ------------------------------------------------------------------ #
    def _init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Element Quest")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_sm  = pygame.font.SysFont('Arial', 16)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self):
        self.grid = [row[:] for row in self._base_grid]
        self.frame = 0
        self.total_gems = sum(
            row.count(int(Tile.FIRE_GEM)) + row.count(int(Tile.WATER_GEM))
            for row in self.grid
        )

        fc, fr = self.fire_spawn
        self.fire_char = Character(CharacterType.FIRE, fc, fr)

        if self.agents == 'both':
            wc, wr = self.water_spawn
            self.water_char = Character(CharacterType.WATER, wc, wr)
        else:
            self.water_char = None

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, fire_action: int = 0, water_action: int = 0) -> dict:
        """
        Advance one physics frame.
        Returns a dict with events that occurred this frame.
        """
        all_events: dict = {}

        if self.fire_char and self.fire_char.alive and not self.fire_char.at_door:
            self.fire_char.apply_action(fire_action)
            ev = self.fire_char.update(self.grid)
            all_events['fire'] = ev
            if 'collect_gem' in ev:
                col, row, _ = ev['collect_gem']
                self.grid[row][col] = int(Tile.EMPTY)

        if self.water_char and self.water_char.alive and not self.water_char.at_door:
            self.water_char.apply_action(water_action)
            ev = self.water_char.update(self.grid)
            all_events['water'] = ev
            if 'collect_gem' in ev:
                col, row, _ = ev['collect_gem']
                self.grid[row][col] = int(Tile.EMPTY)

        self.frame += 1
        return all_events

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def is_done(self) -> bool:
        fire_done = (self.fire_char is None or
                     not self.fire_char.alive or
                     self.fire_char.at_door)
        if self.agents == 'both':
            water_done = (self.water_char is None or
                          not self.water_char.alive or
                          self.water_char.at_door)
            return fire_done and water_done
        return fire_done

    def fire_alive(self) -> bool:
        return self.fire_char is not None and self.fire_char.alive

    def water_alive(self) -> bool:
        return self.water_char is not None and self.water_char.alive

    def gem_positions(self, gem_tile: int) -> list[tuple[int, int]]:
        return [(c, r)
                for r, row in enumerate(self.grid)
                for c, t in enumerate(row)
                if t == gem_tile]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != 'human':
            return
        self.screen.fill(BG_COLOR)
        self._draw_ui()
        self._draw_grid()
        if self.fire_char:
            self.fire_char.draw(self.screen, self.frame)
        if self.water_char:
            self.water_char.draw(self.screen, self.frame)
        pygame.display.flip()

    def _draw_grid(self):
        for r, row in enumerate(self.grid):
            for c, tile in enumerate(row):
                if tile == int(Tile.EMPTY):
                    continue
                rect = pygame.Rect(c * TILE_SIZE,
                                   r * TILE_SIZE + UI_HEIGHT,
                                   TILE_SIZE, TILE_SIZE)
                draw_tile(self.screen, tile, rect, self.frame)

    def _draw_ui(self):
        # Background bar
        pygame.draw.rect(self.screen, (20, 20, 35), (0, 0, WINDOW_W, UI_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 80), (0, UI_HEIGHT), (WINDOW_W, UI_HEIGHT), 2)

        # Fire side
        fire_gems = self.fire_char.gems if self.fire_char else 0
        fire_alive = self.fire_char.alive if self.fire_char else False
        fire_door  = self.fire_char.at_door if self.fire_char else False
        fire_status = "DONE" if fire_door else ("DEAD" if not fire_alive else "")

        pygame.draw.circle(self.screen, (220, 60, 10), (28, 35), 16)
        txt = self.font_big.render(f"FireAgent  Gems: {fire_gems}  {fire_status}", True, (255, 140, 60))
        self.screen.blit(txt, (54, 22))

        # Water side (if active)
        if self.water_char:
            water_gems = self.water_char.gems
            water_alive = self.water_char.alive
            water_door  = self.water_char.at_door
            water_status = "DONE" if water_door else ("DEAD" if not water_alive else "")
            pygame.draw.circle(self.screen, (20, 100, 210), (WINDOW_W - 28, 35), 16)
            txt2 = self.font_big.render(
                f"WaterAgent  Gems: {water_gems}  {water_status}", True, (60, 160, 255))
            self.screen.blit(txt2, (WINDOW_W // 2 + 10, 22))

        # Frame counter
        frame_txt = self.font_sm.render(f"Frame: {self.frame}", True, (120, 120, 140))
        self.screen.blit(frame_txt, (WINDOW_W // 2 - 40, 50))

    def tick(self, fps: int = 60):
        if self.render_mode == 'human':
            self.clock.tick(fps)

    def close(self):
        if self.render_mode == 'human' and pygame.get_init():
            pygame.quit()
