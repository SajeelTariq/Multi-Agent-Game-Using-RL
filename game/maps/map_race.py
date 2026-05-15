"""
Race map — FireAgent (human/AI) vs WaterAgent (AI/human).
Both share the same map; each has their own gems and door.
FireAgent spawns left, WaterAgent spawns right.
"""
from game.tiles import Tile

RAW = [
    "####################",
    "#F               W #",
    "####  ..    ..  ####",
    "#  f w    f w      #",
    "#  ..  ^^  ~~  ..  #",
    "#     ^^^^~~~~     #",
    "# ..   ^^  ~~   .. #",
    "#  f w    f  w  f w#",
    "#  ....  ....  .... #",
    "# ^^^  fw   fw  ~~~#",
    "# ..  ....  ....   #",
    "#  f w   f w   f w #",
    "D....................E",
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
    fire_spawn = (1, 1)
    water_spawn = (18, 1)

    for r, row in enumerate(RAW):
        tile_row = []
        for c, ch in enumerate(row):
            if ch == 'F':
                fire_spawn = (c, r)
            elif ch == 'W':
                water_spawn = (c, r)
            tile_row.append(int(_CHAR_MAP.get(ch, Tile.EMPTY)))
        while len(tile_row) < 20:
            tile_row.append(int(Tile.EMPTY))
        grid.append(tile_row[:20])

    return grid, fire_spawn, water_spawn
