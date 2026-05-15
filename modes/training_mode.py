"""
Mode 1 — Watch a trained agent play.
Loads a saved model and runs it in the training map with rendering.
"""
import pygame
import sys
from pathlib import Path

from rl.env import ElementQuestEnv
from game.maps import map_training


def run(algo_name: str = "dqn"):
    from rl.train import load_model, ALGO_CONFIGS

    print(f"[Watch Mode] Loading {algo_name.upper()} model…")
    model = load_model(algo_name)

    env = ElementQuestEnv(map_module=map_training, render_mode="human")
    obs, _ = env.reset()

    clock = pygame.time.Clock()
    running = True
    episode = 0
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} | reward {total_reward:.1f} | gems {info['gems']}")
            total_reward = 0.0
            obs, _ = env.reset()

        clock.tick(60)

    env.close()
