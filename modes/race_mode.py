"""
Mode 2 — Human vs AI Race.

Human   → controls FireAgent  (WASD or Arrow keys)
AI      → controls WaterAgent (loaded trained model, default DQN)

Both on the same shared race map. First to reach their door with
most gems wins. Score = gems * 10 + (winner bonus 50).
"""
import sys
import pygame
import numpy as np
from pathlib import Path

from game.engine import GameEngine
from game.maps import map_race
from game.tiles import (
    Tile, TILE_SIZE, GRID_COLS, GRID_ROWS,
    UI_HEIGHT, WINDOW_W, WINDOW_H, BG_COLOR,
)
from game.characters import (
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP,
    CharacterType,
)
from rl.env import ElementQuestEnv   # for obs builder
from rl.train import load_model


# ------------------------------------------------------------------ #
# Build observation for WaterAgent (mirrors ElementQuestEnv._get_obs)
# ------------------------------------------------------------------ #
def _water_obs(engine: GameEngine) -> np.ndarray:
    wc   = engine.water_char
    grid = engine.grid

    norm_x  = wc.px / (GRID_COLS * TILE_SIZE)
    norm_y  = wc.py / (GRID_ROWS * TILE_SIZE)
    norm_vx = np.clip(wc.vx / 10.0, -1, 1)
    norm_vy = np.clip(wc.vy / 16.0, -1, 1)
    on_gnd  = float(wc.on_ground)

    cx = wc.px + 16
    cy = wc.py + 20

    gem_pos = engine.gem_positions(int(Tile.WATER_GEM))
    if gem_pos:
        nearest = min(gem_pos,
                      key=lambda g: abs(cx - g[0]*TILE_SIZE) + abs(cy - g[1]*TILE_SIZE))
        gem_dx = np.clip((nearest[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
        gem_dy = np.clip((nearest[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
    else:
        gem_dx, gem_dy = 0.0, 0.0

    door_pos = engine.gem_positions(int(Tile.WATER_DOOR))
    if door_pos:
        d = door_pos[0]
        door_dx = np.clip((d[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
        door_dy = np.clip((d[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
    else:
        door_dx, door_dy = 0.0, 0.0

    total_w = sum(row.count(int(Tile.WATER_GEM)) for row in engine._base_grid)
    gems_remaining = len(gem_pos) / max(total_w, 1)

    col_c = int(cx // TILE_SIZE)
    row_c = int(cy // TILE_SIZE)
    local = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = row_c + dr, col_c + dc
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                local.append(grid[r][c] / 9.0)
            else:
                local.append(1.0)

    return np.array([
        norm_x, norm_y, norm_vx, norm_vy, on_gnd,
        gem_dx, gem_dy, door_dx, door_dy, gems_remaining,
        *local,
    ], dtype=np.float32)


# ------------------------------------------------------------------ #
def _keyboard_action() -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]  or keys[pygame.K_a]:
        return ACTION_LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return ACTION_RIGHT
    if keys[pygame.K_UP]    or keys[pygame.K_w] or keys[pygame.K_SPACE]:
        return ACTION_JUMP
    return ACTION_NOOP


# ------------------------------------------------------------------ #
def run(algo_name: str = "dqn"):
    print(f"[Race Mode] Loading AI model ({algo_name.upper()})…")
    ai_model = load_model(algo_name)

    engine = GameEngine(map_race, agents='both', render_mode='human')
    engine.reset()

    pygame.init()
    font    = pygame.font.SysFont('Arial', 18, bold=True)
    font_lg = pygame.font.SysFont('Arial', 38, bold=True)
    clock   = pygame.time.Clock()

    FRAME_SKIP  = 2   # smoother AI
    frame_count = 0
    ai_action   = ACTION_NOOP
    result_msg  = ""
    result_timer = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and result_msg:
                    engine.reset()
                    result_msg   = ""
                    result_timer = 0

        if not result_msg:
            human_action = _keyboard_action()

            # AI decision every FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0 and engine.water_char.alive:
                w_obs    = _water_obs(engine)
                ai_action, _ = ai_model.predict(w_obs, deterministic=True)

            engine.step(fire_action=human_action, water_action=int(ai_action))
            frame_count += 1

            # Check end conditions
            fc = engine.fire_char
            wc = engine.water_char

            fire_done  = fc.at_door or not fc.alive
            water_done = wc.at_door or not wc.alive

            if fire_done or water_done:
                fire_score  = fc.gems * 10  + (50 if fc.at_door  else 0)
                water_score = wc.gems * 10  + (50 if wc.at_door  else 0)

                if fire_score > water_score:
                    result_msg = f"YOU WIN!  {fire_score} vs {water_score}"
                elif water_score > fire_score:
                    result_msg = f"AI WINS!  {water_score} vs {fire_score}"
                else:
                    result_msg = f"DRAW!  {fire_score} each"
                result_timer = 180  # show for 3 seconds then prompt

        engine.render()

        # HUD overlay: controls reminder
        hint = font.render("Arrow/WASD: Move  |  W/Space: Jump  |  R: Restart  |  ESC: Quit",
                           True, (160, 160, 180))
        engine.screen.blit(hint, (10, WINDOW_H - 28))

        # Result overlay
        if result_msg:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            engine.screen.blit(overlay, (0, 0))
            txt = font_lg.render(result_msg, True, (255, 220, 60))
            engine.screen.blit(txt, txt.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 30)))
            sub = font.render("Press R to restart or ESC to quit", True, (200, 200, 200))
            engine.screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 30)))

        pygame.display.flip()
        clock.tick(60)

    engine.close()
