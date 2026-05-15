"""
Race map — FireAgent (AI) vs WaterAgent (Human).
Shares structural feel with map_training: scattered platforms,
gems on platforms, hazards as floor-level pools.

Doors are at row 11 (clearly visible), solid floor at row 12.
FireAgent door (D) bottom-left, WaterAgent door (E) bottom-right.

Hazard lanes:
  Left-centre  → fire hazards (^^^) — safe for FireAgent, lethal for WaterAgent
  Right-centre → water hazards (~~~) — safe for WaterAgent, lethal for FireAgent
  Both sides always have a passable gap so neither character is trapped.
"""
from game.tiles import Tile

RAW = [
    "####################",  # 0  top wall
    "#F               W #",  # 1  spawns: FireAgent col-1, WaterAgent col-17
    "#####  ..    ..  ###",  # 2  wide platforms from each wall
    "#  f w    f  w     #",  # 3  first gems
    "#  ..  ^^^  ~~~  . #",  # 4  platforms + hazard pools (gaps on sides)
    "#    ^^^     ~~~    #",  # 5  hazard zone — empty cols 1-4 and 16-19 passable
    "# ..   ^^   ~~   ..#",  # 6  platforms on each side, hazards in centre
    "#  f w    f w  f w #",  # 7  gems across the map
    "#  ...  ....  ...  #",  # 8  platforms (similar spread to training map)
    "# ^^  f w   f w  ~~#",  # 9  hazards close to walls + gems
    "# ..  ....   ....  #",  # 10 platforms (mirror of training map row 10)
    "D  f w   f w   f w E",  # 11 DOORS at col-0 (Fire) and col-19 (Water) + gems
    "####################",  # 12 solid bottom wall — characters walk on this
]

_CHAR_MAP = {
    '#': Tile.WALL,
    '.': Tile.PLATFORM,
    '^': Tile.FIRE_HAZARD,
    '~': Tile.WATER_HAZARD,
    '@': Tile.ACID,
    'f': Tile.FIRE_GEM,
    'w': Tile.WATER_GEM,
    'D': Tile.FIRE_DOOR,
    'E': Tile.WATER_DOOR,
    ' ': Tile.EMPTY,
    'F': Tile.EMPTY,
    'W': Tile.EMPTY,
}


def load():
    """Return (grid, fire_spawn, water_spawn)."""
    grid = []
    fire_spawn  = (1, 1)
    water_spawn = (17, 1)

    for r, row_str in enumerate(RAW):
        tile_row = []
        for c, ch in enumerate(row_str):
            if ch == 'F':
                fire_spawn = (c, r)
            elif ch == 'W':
                water_spawn = (c, r)
            tile_row.append(int(_CHAR_MAP.get(ch, Tile.EMPTY)))
        # safety: pad / trim to exactly 20 columns
        while len(tile_row) < 20:
            tile_row.append(int(Tile.EMPTY))
        grid.append(tile_row[:20])

    return grid, fire_spawn, water_spawn
