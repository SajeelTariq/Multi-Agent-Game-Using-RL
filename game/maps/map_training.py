"""
Training map — single FireAgent.
20 cols x 13 rows  (GRID_COLS x GRID_ROWS).

Legend:
  #  WALL
  .  PLATFORM (walkable)
  ^  FIRE_HAZARD  (kills WaterAgent, safe for FireAgent)
  ~  WATER_HAZARD (kills FireAgent)
  @  ACID          (kills both)
  f  FIRE_GEM
  w  WATER_GEM
  F  FireAgent spawn  (rendered as EMPTY)
  W  WaterAgent spawn (rendered as EMPTY)
  D  FIRE_DOOR
  E  WATER_DOOR
  (space) EMPTY
"""
from game.tiles import Tile

RAW = [
    "####################",
    "#                  #",
    "#F  f              #",
    "####   ..          #",
    "#        f    ...  #",
    "#  ..  ......      #",
    "#        ~~~   f   #",
    "#  ....        ... #",
    "#      f  ..       #",
    "#  .. ....  ~~~    #",
    "#         f    ..  #",
    "#  f  ..        f  D",
    "####################",
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
    fire_spawn = (1, 2)
    water_spawn = (18, 2)

    for r, row in enumerate(RAW):
        tile_row = []
        for c, ch in enumerate(row):
            if ch == 'F':
                fire_spawn = (c, r)
            elif ch == 'W':
                water_spawn = (c, r)
            tile_row.append(int(_CHAR_MAP.get(ch, Tile.EMPTY)))
        # pad / trim to GRID_COLS
        while len(tile_row) < 20:
            tile_row.append(int(Tile.EMPTY))
        grid.append(tile_row[:20])

    return grid, fire_spawn, water_spawn
