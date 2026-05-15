from __future__ import annotations
import pygame
from game.tiles import (
    TILE_SIZE, CHAR_WIDTH, CHAR_HEIGHT, UI_HEIGHT,
    Tile, SOLID_TILES, LETHAL_TO_FIRE, LETHAL_TO_WATER
)

GRAVITY = 0.7
JUMP_VEL = -14.0
MOVE_SPEED = 4.0
MAX_FALL = 16.0

ACTION_NOOP  = 0
ACTION_LEFT  = 1
ACTION_RIGHT = 2
ACTION_JUMP  = 3


class CharacterType:
    FIRE  = "fire"
    WATER = "water"


class Character:
    def __init__(self, char_type: str, spawn_col: int, spawn_row: int):
        self.char_type = char_type
        self.spawn_col = spawn_col
        self.spawn_row = spawn_row
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.px = float(self.spawn_col * TILE_SIZE + (TILE_SIZE - CHAR_WIDTH) // 2)
        self.py = float(self.spawn_row * TILE_SIZE + (TILE_SIZE - CHAR_HEIGHT))
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.alive = True
        self.at_door = False
        self.gems = 0
        self.facing = 1  # 1 = right, -1 = left

    # ------------------------------------------------------------------ #
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.px), int(self.py) + UI_HEIGHT,
                           CHAR_WIDTH, CHAR_HEIGHT)

    @property
    def game_rect(self) -> pygame.Rect:
        """Rect in game-space (no UI offset) for collision."""
        return pygame.Rect(int(self.px), int(self.py), CHAR_WIDTH, CHAR_HEIGHT)

    # ------------------------------------------------------------------ #
    def apply_action(self, action: int):
        if not self.alive or self.at_door:
            return
        if action == ACTION_LEFT:
            self.vx = -MOVE_SPEED
            self.facing = -1
        elif action == ACTION_RIGHT:
            self.vx = MOVE_SPEED
            self.facing = 1
        elif action == ACTION_JUMP and self.on_ground:
            self.vy = JUMP_VEL
            self.on_ground = False
        else:
            self.vx = 0.0

    # ------------------------------------------------------------------ #
    def update(self, grid: list[list[int]]) -> dict:
        """Simulate one physics frame. Returns event dict."""
        if not self.alive or self.at_door:
            return {}

        # Gravity
        self.vy = min(self.vy + GRAVITY, MAX_FALL)

        events = {}

        # --- Horizontal movement ---
        self.px += self.vx
        self._resolve_horizontal(grid)

        # --- Vertical movement ---
        self.py += self.vy
        self._resolve_vertical(grid)

        # --- Hazard / gem / door checks ---
        self._check_tiles(grid, events)

        return events

    # ------------------------------------------------------------------ #
    def _tile_at(self, grid, px, py):
        col = int(px // TILE_SIZE)
        row = int(py // TILE_SIZE)
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return Tile(grid[row][col])
        return Tile.WALL

    def _is_solid(self, grid, px, py):
        return self._tile_at(grid, px, py) in SOLID_TILES

    # ------------------------------------------------------------------ #
    def _resolve_horizontal(self, grid):
        r = self.game_rect
        margin = 2
        if self.vx > 0:
            right = self.px + CHAR_WIDTH
            if (self._is_solid(grid, right, self.py + margin) or
                    self._is_solid(grid, right, self.py + CHAR_HEIGHT - margin)):
                self.px = (int(right // TILE_SIZE)) * TILE_SIZE - CHAR_WIDTH
                self.vx = 0
        elif self.vx < 0:
            left = self.px
            if (self._is_solid(grid, left, self.py + margin) or
                    self._is_solid(grid, left, self.py + CHAR_HEIGHT - margin)):
                self.px = (int(left // TILE_SIZE) + 1) * TILE_SIZE
                self.vx = 0

    def _resolve_vertical(self, grid):
        margin = 2
        if self.vy > 0:
            bottom = self.py + CHAR_HEIGHT
            if (self._is_solid(grid, self.px + margin, bottom) or
                    self._is_solid(grid, self.px + CHAR_WIDTH - margin, bottom)):
                self.py = (int(bottom // TILE_SIZE)) * TILE_SIZE - CHAR_HEIGHT
                self.vy = 0
                self.on_ground = True
        elif self.vy < 0:
            top = self.py
            if (self._is_solid(grid, self.px + margin, top) or
                    self._is_solid(grid, self.px + CHAR_WIDTH - margin, top)):
                self.py = (int(top // TILE_SIZE) + 1) * TILE_SIZE
                self.vy = 0
        else:
            # Check still on ground
            bottom = self.py + CHAR_HEIGHT + 1
            if not (self._is_solid(grid, self.px + margin, bottom) or
                    self._is_solid(grid, self.px + CHAR_WIDTH - margin, bottom)):
                self.on_ground = False

    # ------------------------------------------------------------------ #
    def _check_tiles(self, grid: list[list[int]], events: dict):
        cx = self.px + CHAR_WIDTH // 2
        cy = self.py + CHAR_HEIGHT // 2

        tile = self._tile_at(grid, cx, cy)
        foot = self._tile_at(grid, cx, self.py + CHAR_HEIGHT - 2)

        for t in [tile, foot]:
            if t == Tile.FIRE_GEM and self.char_type == CharacterType.FIRE:
                col = int(cx // TILE_SIZE)
                row = int((self.py + CHAR_HEIGHT - 2) // TILE_SIZE)
                events['collect_gem'] = (col, row, Tile.FIRE_GEM)
                self.gems += 1
                break
            if t == Tile.WATER_GEM and self.char_type == CharacterType.WATER:
                col = int(cx // TILE_SIZE)
                row = int((self.py + CHAR_HEIGHT - 2) // TILE_SIZE)
                events['collect_gem'] = (col, row, Tile.WATER_GEM)
                self.gems += 1
                break

        # Lethal tile check
        lethal = LETHAL_TO_FIRE if self.char_type == CharacterType.FIRE else LETHAL_TO_WATER
        if tile in lethal or foot in lethal:
            self.alive = False
            events['died'] = True

        # Door check
        door = Tile.FIRE_DOOR if self.char_type == CharacterType.FIRE else Tile.WATER_DOOR
        if tile == door or foot == door:
            self.at_door = True
            events['reached_door'] = True

        # Out of bounds
        if self.py > len(grid) * TILE_SIZE:
            self.alive = False
            events['died'] = True

    # ------------------------------------------------------------------ #
    def draw(self, surface: pygame.Surface, frame: int = 0):
        if not self.alive:
            return
        r = self.rect
        if self.char_type == CharacterType.FIRE:
            self._draw_fire_char(surface, r, frame)
        else:
            self._draw_water_char(surface, r, frame)

    def _draw_fire_char(self, surface, r, frame):
        # Body
        body_color = (220, 60, 10)
        pygame.draw.rect(surface, body_color,
                         (r.x + 4, r.y + 12, r.width - 8, r.height - 12), border_radius=4)
        # Head
        pygame.draw.circle(surface, (230, 80, 20), (r.centerx, r.y + 10), 11)
        # Eyes
        eye_x = r.centerx + self.facing * 4
        pygame.draw.circle(surface, (255, 220, 0), (eye_x, r.y + 8), 3)
        pygame.draw.circle(surface, (0, 0, 0), (eye_x + self.facing, r.y + 8), 1)
        # Flame crown
        flicker = abs((frame % 10) - 5)
        for i in range(3):
            ox = r.x + 6 + i * 10
            oy = r.y - 4 - flicker + i % 2 * 2
            pygame.draw.polygon(surface, (255, 140, 0),
                                [(ox, oy + 8), (ox - 4, oy + 12), (ox + 4, oy + 12)])
        # Legs
        leg_y = r.bottom - 8
        step = (frame // 4) % 2
        pygame.draw.rect(surface, (180, 40, 0),
                         (r.x + 4, leg_y, 10, 8 + step * 2))
        pygame.draw.rect(surface, (180, 40, 0),
                         (r.right - 14, leg_y, 10, 8 - step * 2))
        # at_door tint
        if self.at_door:
            s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            s.fill((255, 255, 255, 80))
            surface.blit(s, r.topleft)

    def _draw_water_char(self, surface, r, frame):
        body_color = (20, 100, 210)
        pygame.draw.rect(surface, body_color,
                         (r.x + 4, r.y + 12, r.width - 8, r.height - 12), border_radius=4)
        pygame.draw.circle(surface, (30, 120, 220), (r.centerx, r.y + 10), 11)
        eye_x = r.centerx + self.facing * 4
        pygame.draw.circle(surface, (200, 240, 255), (eye_x, r.y + 8), 3)
        pygame.draw.circle(surface, (0, 0, 0), (eye_x + self.facing, r.y + 8), 1)
        # Water droplet crown
        wave = abs((frame % 12) - 6)
        pygame.draw.circle(surface, (100, 180, 255),
                           (r.centerx, r.y - 4 - wave // 2), 5)
        leg_y = r.bottom - 8
        step = (frame // 4) % 2
        pygame.draw.rect(surface, (15, 70, 170),
                         (r.x + 4, leg_y, 10, 8 + step * 2))
        pygame.draw.rect(surface, (15, 70, 170),
                         (r.right - 14, leg_y, 10, 8 - step * 2))
        if self.at_door:
            s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            s.fill((255, 255, 255, 80))
            surface.blit(s, r.topleft)
