from enum import IntEnum
import pygame

TILE_SIZE = 48
GRID_COLS = 20
GRID_ROWS = 13
CHAR_WIDTH = 32
CHAR_HEIGHT = 40

UI_HEIGHT = 70  # top bar for scores/status
WINDOW_W = GRID_COLS * TILE_SIZE          # 960
WINDOW_H = GRID_ROWS * TILE_SIZE + UI_HEIGHT  # 694


class Tile(IntEnum):
    EMPTY = 0
    PLATFORM = 1
    FIRE_HAZARD = 2   # kills WaterAgent
    WATER_HAZARD = 3  # kills FireAgent
    ACID = 4          # kills both
    FIRE_GEM = 5      # collected by FireAgent
    WATER_GEM = 6     # collected by WaterAgent
    FIRE_DOOR = 7     # FireAgent exit
    WATER_DOOR = 8    # WaterAgent exit
    WALL = 9


SOLID_TILES = {Tile.WALL, Tile.PLATFORM}
LETHAL_TO_FIRE = {Tile.WATER_HAZARD, Tile.ACID}
LETHAL_TO_WATER = {Tile.FIRE_HAZARD, Tile.ACID}

# Base colors used for state encoding and fallback
TILE_COLORS = {
    Tile.EMPTY:       (15, 15, 25),
    Tile.PLATFORM:    (90, 60, 30),
    Tile.FIRE_HAZARD: (220, 70, 10),
    Tile.WATER_HAZARD:(20, 90, 200),
    Tile.ACID:        (30, 180, 30),
    Tile.FIRE_GEM:    (255, 140, 0),
    Tile.WATER_GEM:   (0, 180, 255),
    Tile.FIRE_DOOR:   (180, 30, 30),
    Tile.WATER_DOOR:  (30, 30, 180),
    Tile.WALL:        (50, 50, 60),
}

BG_COLOR = (15, 15, 25)


def draw_tile(surface: pygame.Surface, tile: int, rect: pygame.Rect, frame: int = 0):
    t = Tile(tile)

    if t == Tile.EMPTY:
        return  # background already drawn

    elif t == Tile.WALL:
        pygame.draw.rect(surface, (50, 50, 62), rect)
        pygame.draw.rect(surface, (70, 70, 85), rect, 2)

    elif t == Tile.PLATFORM:
        pygame.draw.rect(surface, (90, 60, 30), rect)
        pygame.draw.rect(surface, (130, 90, 50), (rect.x, rect.y, rect.width, 6))
        pygame.draw.rect(surface, (60, 40, 15), rect, 1)

    elif t == Tile.FIRE_HAZARD:
        flicker = abs((frame % 12) - 6)
        pygame.draw.rect(surface, (180, 40, 0), rect)
        for i in range(3):
            ox = (rect.x + 6 + i * 14)
            oy = rect.y + 4 + (flicker + i * 2) % 10
            pts = [(ox, oy + 20), (ox - 6, oy + 30), (ox + 6, oy + 30)]
            pygame.draw.polygon(surface, (255, 140 + flicker * 5, 0), pts)
        pygame.draw.rect(surface, (255, 80, 0), (rect.x, rect.y + rect.height - 8, rect.width, 8))

    elif t == Tile.WATER_HAZARD:
        wave = abs((frame % 16) - 8)
        pygame.draw.rect(surface, (15, 70, 170), rect)
        pygame.draw.rect(surface, (60, 140, 230), (rect.x, rect.y + wave, rect.width, 5))
        pygame.draw.rect(surface, (100, 180, 255), (rect.x, rect.y + wave + 3, rect.width, 2))

    elif t == Tile.ACID:
        bubble_y = (frame * 2) % rect.height
        pygame.draw.rect(surface, (20, 140, 20), rect)
        pygame.draw.rect(surface, (60, 210, 60), (rect.x, rect.y, rect.width, 4))
        bx = rect.x + 8 + (frame * 3) % (rect.width - 16)
        by = rect.bottom - bubble_y - 4
        if rect.top < by < rect.bottom:
            pygame.draw.circle(surface, (120, 255, 120), (bx, by), 3)

    elif t == Tile.FIRE_GEM:
        cx, cy = rect.centerx, rect.centery
        glow = abs((frame % 20) - 10) * 2
        pts = [(cx, cy - 11), (cx + 8, cy), (cx, cy + 9), (cx - 8, cy)]
        pygame.draw.polygon(surface, (255, 100 + glow, 0), pts)
        inner = [(cx, cy - 7), (cx + 4, cy), (cx, cy + 5), (cx - 4, cy)]
        pygame.draw.polygon(surface, (255, 220, 100), inner)

    elif t == Tile.WATER_GEM:
        cx, cy = rect.centerx, rect.centery
        glow = abs((frame % 20) - 10) * 2
        pts = [(cx, cy - 11), (cx + 8, cy), (cx, cy + 9), (cx - 8, cy)]
        pygame.draw.polygon(surface, (0, 140 + glow, 255), pts)
        inner = [(cx, cy - 7), (cx + 4, cy), (cx, cy + 5), (cx - 4, cy)]
        pygame.draw.polygon(surface, (160, 230, 255), inner)

    elif t == Tile.FIRE_DOOR:
        pygame.draw.rect(surface, (120, 20, 20), rect)
        pygame.draw.rect(surface, (200, 80, 40), rect, 4)
        # arch
        pygame.draw.rect(surface, (160, 40, 40),
                         (rect.x + 6, rect.y + 6, rect.width - 12, rect.height - 6))
        # knob
        pygame.draw.circle(surface, (255, 200, 0),
                           (rect.right - 10, rect.centery), 4)
        # fire label
        pygame.draw.rect(surface, (255, 80, 0),
                         (rect.x + 8, rect.y + 8, rect.width - 16, 6))

    elif t == Tile.WATER_DOOR:
        pygame.draw.rect(surface, (20, 20, 120), rect)
        pygame.draw.rect(surface, (40, 80, 200), rect, 4)
        pygame.draw.rect(surface, (40, 60, 160),
                         (rect.x + 6, rect.y + 6, rect.width - 12, rect.height - 6))
        pygame.draw.circle(surface, (160, 230, 255),
                           (rect.right - 10, rect.centery), 4)
        pygame.draw.rect(surface, (0, 140, 255),
                         (rect.x + 8, rect.y + 8, rect.width - 16, 6))
