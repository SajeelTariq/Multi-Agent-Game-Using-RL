"""
Element Quest — Main Menu
Navigate with UP/DOWN arrows, select with ENTER.
"""
import sys
import pygame
from game.tiles import WINDOW_W, WINDOW_H

MENU_ITEMS = [
    ("Train Agents  (DQN / PPO / A2C)", "train"),
    ("Watch Trained Agent Play",         "watch"),
    ("Human vs AI  Race Mode",           "race"),
    ("Open Training Dashboard",          "dashboard"),
    ("Quit",                             "quit"),
]

ALGO_ITEMS = ["dqn", "ppo", "a2c", "back"]

TITLE_COLOR  = (255, 140, 60)
SELECT_COLOR = (255, 220, 80)
NORMAL_COLOR = (180, 180, 200)
DIM_COLOR    = (80, 80, 100)
ACCENT_FIRE  = (220, 70, 10)
ACCENT_WATER = (20, 100, 210)


def _draw_bg(screen, frame):
    import math
    screen.fill((12, 12, 22))
    for i in range(5):
        r = 60 + i * 40
        x = WINDOW_W // 2 + int(math.sin(frame * 0.01 + i) * 80)
        y = WINDOW_H // 2 + int(math.cos(frame * 0.008 + i) * 60)
        pygame.draw.circle(screen, (20 + i * 4, 15 + i * 3, 35 + i * 5), (x, y), r, 1)


def _pick_algo(screen, clock, font, font_sm):
    """Algo picker runs inside the existing pygame context."""
    selected = 0
    frame = 0
    while True:
        frame += 1
        _draw_bg(screen, frame)

        title = font.render("Select Algorithm", True, TITLE_COLOR)
        screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 180)))

        for i, label in enumerate(ALGO_ITEMS):
            color = SELECT_COLOR if i == selected else NORMAL_COLOR
            txt = font.render(label.upper(), True, color)
            rect = txt.get_rect(center=(WINDOW_W // 2, 280 + i * 60))
            if i == selected:
                pygame.draw.rect(screen, (40, 30, 10),
                                 rect.inflate(40, 16), border_radius=8)
                pygame.draw.rect(screen, SELECT_COLOR,
                                 rect.inflate(40, 16), 2, border_radius=8)
            screen.blit(txt, rect)

        hint = font_sm.render("↑↓ Navigate   Enter Select   ESC Back",
                              True, DIM_COLOR)
        screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 30)))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(ALGO_ITEMS)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(ALGO_ITEMS)
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return ALGO_ITEMS[selected]
                if event.key == pygame.K_ESCAPE:
                    return 'back'


def _reinit_pygame():
    """Re-initialise pygame after a mode that calls pygame.quit() internally."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Element Quest")
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('Arial', 28, bold=True)
    font_sm = pygame.font.SysFont('Arial', 17)
    return screen, clock, font, font_sm


def main():
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Element Quest")
    clock      = pygame.time.Clock()
    font       = pygame.font.SysFont('Arial', 28, bold=True)
    font_sm    = pygame.font.SysFont('Arial', 17)
    font_title = pygame.font.SysFont('Arial', 52, bold=True)

    selected = 0
    frame = 0

    while True:
        frame += 1
        _draw_bg(screen, frame)

        # Title
        t1 = font_title.render("ELEMENT", True, ACCENT_FIRE)
        t2 = font_title.render("QUEST",   True, ACCENT_WATER)
        w_total = t1.get_width() + 14 + t2.get_width()
        x_start = (WINDOW_W - w_total) // 2
        screen.blit(t1, (x_start, 80))
        screen.blit(t2, (x_start + t1.get_width() + 14, 80))

        sub = font_sm.render("Self-Learning Game AI  ·  DQN vs PPO vs A2C",
                             True, (130, 130, 160))
        screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, 148)))

        pygame.draw.line(screen, (50, 50, 70),
                         (WINDOW_W // 4, 168), (3 * WINDOW_W // 4, 168), 1)

        for i, (label, _) in enumerate(MENU_ITEMS):
            color = SELECT_COLOR if i == selected else NORMAL_COLOR
            if label == "Quit":
                color = (160, 80, 80) if i != selected else (220, 100, 100)
            txt  = font.render(label, True, color)
            rect = txt.get_rect(center=(WINDOW_W // 2, 220 + i * 70))
            if i == selected:
                bg_col = (40, 30, 10) if label != "Quit" else (40, 10, 10)
                pygame.draw.rect(screen, bg_col, rect.inflate(50, 18), border_radius=8)
                pygame.draw.rect(screen, color,  rect.inflate(50, 18), 2, border_radius=8)
            screen.blit(txt, rect)

        hint = font_sm.render("↑↓ Navigate   Enter Select   ESC Quit",
                              True, DIM_COLOR)
        screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 30)))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(MENU_ITEMS)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(MENU_ITEMS)
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    action = MENU_ITEMS[selected][1]

                    if action == "quit":
                        pygame.quit()
                        sys.exit()

                    elif action == "train":
                        # Pick algo inside current pygame context, THEN quit for training
                        algo = _pick_algo(screen, clock, font, font_sm)
                        if algo != 'back':
                            import config as cfg
                            print(f"\nStarting training: {algo.upper()}")
                            print("Tip: run 'python dashboard/app.py' in another terminal for live charts.")
                            if cfg.RENDER_TRAINING:
                                print("     Continuous render window is ON (set RENDER_TRAINING=False in config.py to speed up).")
                            elif cfg.RENDER_PREVIEW_EVERY > 0:
                                print(f"     Preview window every {cfg.RENDER_PREVIEW_EVERY} episodes.")
                            pygame.quit()
                            from rl.train import train_algo
                            train_algo(algo)
                            screen, clock, font, font_sm = _reinit_pygame()
                            font_title = pygame.font.SysFont('Arial', 52, bold=True)

                    elif action == "watch":
                        algo = _pick_algo(screen, clock, font, font_sm)
                        if algo != 'back':
                            pygame.quit()
                            try:
                                from modes.training_mode import run
                                run(algo)
                            except FileNotFoundError as e:
                                print(f"\nError: {e}")
                            screen, clock, font, font_sm = _reinit_pygame()
                            font_title = pygame.font.SysFont('Arial', 52, bold=True)

                    elif action == "race":
                        algo = _pick_algo(screen, clock, font, font_sm)
                        if algo != 'back':
                            pygame.quit()
                            try:
                                from modes.race_mode import run
                                run(algo)
                            except FileNotFoundError as e:
                                print(f"\nError: {e}")
                            screen, clock, font, font_sm = _reinit_pygame()
                            font_title = pygame.font.SysFont('Arial', 52, bold=True)

                    elif action == "dashboard":
                        import subprocess
                        subprocess.Popen([sys.executable, "dashboard/app.py"])
                        print("\nDashboard starting at http://127.0.0.1:8050")


if __name__ == "__main__":
    main()
