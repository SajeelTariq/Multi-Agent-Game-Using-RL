"""
Mode 2 — Human vs AI Race.

Human → controls WaterAgent  (arrow keys / WASD)
AI    → controls FireAgent   (loaded trained model — same character it was trained on)

Both on the same shared race map. Each collects their own gems and races to their door.

Win rules:
  - Reaching your door = WIN for that side (instant win if other hasn't reached yet)
  - Dying = LOSE for that side (instant loss, other side wins)
  - If neither reaches door (both die or time up): higher score wins
  Score = gems * 10 + 50 door bonus
"""
import sys
import pygame
import numpy as np

from game.engine import GameEngine
from game.maps import map_race
from game.tiles import (
    Tile, TILE_SIZE, GRID_COLS, GRID_ROWS,
    UI_HEIGHT, WINDOW_W, WINDOW_H, BG_COLOR,
)
from game.characters import (
    ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP,
)
from rl.train import load_model

MAX_RACE_STEPS = 3000  # ~50 seconds at 60fps


# ------------------------------------------------------------------ #
# FireAgent observation (same format as rl/env.py — what model was trained on)
# ------------------------------------------------------------------ #
def _fire_obs(engine: GameEngine) -> np.ndarray:
    fc   = engine.fire_char
    grid = engine.grid

    norm_x  = fc.px / (GRID_COLS * TILE_SIZE)
    norm_y  = fc.py / (GRID_ROWS * TILE_SIZE)
    norm_vx = np.clip(fc.vx / 10.0, -1, 1)
    norm_vy = np.clip(fc.vy / 16.0, -1, 1)
    on_gnd  = float(fc.on_ground)

    cx = fc.px + 16
    cy = fc.py + 20

    gem_pos = engine.gem_positions(int(Tile.FIRE_GEM))
    if gem_pos:
        nearest = min(gem_pos,
                      key=lambda g: abs(cx - g[0]*TILE_SIZE) + abs(cy - g[1]*TILE_SIZE))
        gem_dx = np.clip((nearest[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
        gem_dy = np.clip((nearest[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
    else:
        gem_dx, gem_dy = 0.0, 0.0

    door_pos = engine.gem_positions(int(Tile.FIRE_DOOR))
    if door_pos:
        d = door_pos[0]
        door_dx = np.clip((d[0]*TILE_SIZE - cx) / (GRID_COLS*TILE_SIZE), -1, 1)
        door_dy = np.clip((d[1]*TILE_SIZE - cy) / (GRID_ROWS*TILE_SIZE), -1, 1)
    else:
        door_dx, door_dy = 0.0, 0.0

    total_f = sum(row.count(int(Tile.FIRE_GEM)) for row in engine._base_grid)
    gems_remaining = len(gem_pos) / max(total_f, 1)

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
def _resolve_result(fc, wc, timed_out: bool) -> str:
    """
    Return a result message string based on character states.
    Human = WaterAgent (wc), AI = FireAgent (fc).
    """
    ai_won   = fc.at_door
    human_won = wc.at_door
    ai_dead   = not fc.alive
    human_dead = not wc.alive

    ai_score    = fc.gems * 10 + (50 if fc.at_door else 0)
    human_score = wc.gems * 10 + (50 if wc.at_door else 0)

    # Door reached — door wins over death/score
    if human_won and not ai_won:
        return f"YOU WIN!   {human_score} pts  vs  AI {ai_score} pts"
    if ai_won and not human_won:
        return f"AI WINS!   AI {ai_score} pts  vs  {human_score} pts"
    if human_won and ai_won:
        if human_score >= ai_score:
            return f"YOU WIN!   {human_score} pts  vs  AI {ai_score} pts"
        return f"AI WINS!   AI {ai_score} pts  vs  {human_score} pts"

    # Deaths (neither reached door)
    if human_dead and not ai_dead:
        return f"AI WINS!  You died.  AI {ai_score} pts  vs  {human_score} pts"
    if ai_dead and not human_dead:
        return f"YOU WIN!  AI died.  {human_score} pts  vs  AI {ai_score} pts"
    if human_dead and ai_dead:
        if human_score > ai_score:
            return f"YOU WIN (on points)!  {human_score} vs {ai_score}"
        if ai_score > human_score:
            return f"AI WINS (on points)!  AI {ai_score} vs {human_score}"
        return "DRAW!  Both died with equal score."

    # Time up
    if timed_out:
        if human_score > ai_score:
            return f"TIME UP — YOU WIN!   {human_score} pts  vs  AI {ai_score} pts"
        if ai_score > human_score:
            return f"TIME UP — AI WINS!   AI {ai_score} pts  vs  {human_score} pts"
        return f"TIME UP — DRAW!   {human_score} pts each"

    return ""


# ------------------------------------------------------------------ #
def run(algo_name: str = "dqn"):
    print(f"[Race Mode] Loading AI model ({algo_name.upper()})…")
    ai_model = load_model(algo_name)

    engine = GameEngine(map_race, agents='both', render_mode='human')
    engine.reset()

    pygame.init()
    font    = pygame.font.SysFont('Arial', 18, bold=True)
    font_lg = pygame.font.SysFont('Arial', 36, bold=True)
    font_sm = pygame.font.SysFont('Arial', 14)
    clock   = pygame.time.Clock()

    FRAME_SKIP  = 2
    frame_count = 0
    ai_action   = ACTION_NOOP
    result_msg  = ""
    timed_out   = False

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
                    result_msg  = ""
                    timed_out   = False
                    frame_count = 0
                    ai_action   = ACTION_NOOP

        fc = engine.fire_char   # AI
        wc = engine.water_char  # Human

        if not result_msg:
            # Human controls WaterAgent
            human_action = _keyboard_action()

            # AI controls FireAgent — every FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0 and fc.alive and not fc.at_door:
                obs       = _fire_obs(engine)
                ai_action, _ = ai_model.predict(obs, deterministic=True)

            engine.step(fire_action=int(ai_action), water_action=human_action)
            frame_count += 1

            # --- Win / lose conditions ---
            fire_done  = fc.at_door or not fc.alive
            water_done = wc.at_door or not wc.alive

            if fire_done or water_done:
                result_msg = _resolve_result(fc, wc, timed_out=False)
            elif frame_count >= MAX_RACE_STEPS:
                timed_out  = True
                result_msg = _resolve_result(fc, wc, timed_out=True)

        engine.render()

        # --- Labels: who is who ---
        label_ai    = font_sm.render("◀ AI (FireAgent)", True, (255, 140, 60))
        label_human = font_sm.render("WaterAgent (You) ▶", True, (80, 180, 255))
        engine.screen.blit(label_ai,    (6,  UI_HEIGHT + 4))
        engine.screen.blit(label_human, (WINDOW_W - label_human.get_width() - 6,
                                         UI_HEIGHT + 4))

        # --- Controls hint ---
        hint = font_sm.render(
            "Arrow/WASD: Move   W/Space: Jump   R: Restart   ESC: Quit",
            True, (120, 120, 150))
        engine.screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2,
                                   WINDOW_H - 24))

        # --- Result overlay ---
        if result_msg:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            engine.screen.blit(overlay, (0, 0))

            txt = font_lg.render(result_msg, True, (255, 220, 60))
            engine.screen.blit(txt, txt.get_rect(
                center=(WINDOW_W // 2, WINDOW_H // 2 - 30)))

            sub = font.render("Press R to play again  |  ESC to quit",
                               True, (200, 200, 200))
            engine.screen.blit(sub, sub.get_rect(
                center=(WINDOW_W // 2, WINDOW_H // 2 + 20)))

        pygame.display.flip()
        clock.tick(60)

    engine.close()
