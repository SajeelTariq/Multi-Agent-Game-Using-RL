# Element Quest — Self-Learning Game AI

A Fireboy-and-Watergirl-inspired platformer where RL agents (DQN, PPO, A2C) learn
to navigate hazards, collect gems, and race to the exit — with a live training
dashboard and a Human vs AI race mode.

---

## Project Overview

| Item | Detail |
|---|---|
| Game | 2-character grid-based platformer (FireAgent + WaterAgent) |
| RL Algorithms | DQN · PPO · A2C (Stable-Baselines3) |
| State | Feature vector: positions, velocity, nearby tiles, gem/door directions |
| Actions | Left · Right · Jump · No-op (Discrete 4) |
| Reward | +10 gem · +150 win · −100 death · −0.05/step |
| Dashboard | Dash/Plotly — live reward, episode length, algorithm comparison |
| Human vs AI | Race map — keyboard vs trained AI, shared map, score comparison |
| GPU Required | No — MLP networks train fast on CPU |

---

## Modes

### Mode 1 — Train Agents
Train any of the three algorithms. Logs are written to `training_logs/` in real time.

### Mode 2 — Watch Trained Agent
Load a saved model and watch it play the training map with full rendering.

### Mode 3 — Human vs AI Race
You control FireAgent (arrow keys / WASD). The AI controls WaterAgent.
Shared map, separate hazards and gems. Most points + reaching door first = winner.

### Dashboard
Open in browser while training runs. Shows reward curves, episode lengths,
and a final comparison bar chart across all three algorithms.

---

## Setup

### 1. Clone / open the project
```
cd Multi-Agent-Game-Using-RL
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Mac / Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> PyTorch is required by Stable-Baselines3. If the above installs a CPU-only
> torch that is fine — no GPU needed.

---

## Running the Project

### Launch the main menu
```bash
python main.py
```
Navigate with **↑ ↓** arrow keys, select with **Enter**.

---

### Train from the command line (alternative)
```bash
# Train a single algorithm
python rl/train.py --algo dqn
python rl/train.py --algo ppo
python rl/train.py --algo a2c

# Train all three sequentially
python rl/train.py --algo all

# Show a live Pygame preview window every 100 training episodes
python rl/train.py --algo dqn --render-every 100

# Wipe all training logs and saved models (fresh start)
python rl/train.py --reset
```
Saved models appear in `rl/models/`. Training logs go to `training_logs/`.

> **Live preview during training:** Every N episodes the training pauses, opens
> a Pygame window and runs one full episode with the current model so you can
> watch progress. Close the window or press **ESC** to skip and resume training.
> From the main menu this is automatically set to every 100 episodes.

---

### Open the live dashboard (while training runs)
```bash
# In a second terminal
python dashboard/app.py
```
Then open **http://127.0.0.1:8050** in your browser.

> **Reset for a fresh demo:** Click the **"Clear All Logs & Models"** button in
> the dashboard, or run `python rl/train.py --reset` from the terminal.
> Both delete all `.csv` log files and `.zip` model weights so you can start clean.

---

### Watch a trained agent play
```bash
# From the main menu → "Watch Trained Agent Play" → pick algorithm
# Or directly:
python -c "from modes.training_mode import run; run('dqn')"
```

---

### Human vs AI race mode
```bash
# From main menu → "Human vs AI Race Mode" → pick AI algorithm
# Controls:
#   Arrow keys or WASD  → move FireAgent
#   W / Space           → jump
#   R                   → restart after round ends
#   ESC                 → quit
```

---

## Project Structure

```
Multi-Agent-Game-Using-RL/
├── main.py                  ← Main menu (start here)
├── requirements.txt
│
├── game/
│   ├── tiles.py             ← Tile types, constants, drawing
│   ├── characters.py        ← FireAgent & WaterAgent with physics
│   ├── engine.py            ← Core game loop (headless + render modes)
│   └── maps/
│       ├── map_training.py  ← Single-agent training map
│       └── map_race.py      ← Human vs AI shared race map
│
├── rl/
│   ├── env.py               ← Gymnasium environment wrapper
│   ├── train.py             ← DQN / PPO / A2C training + CSV logging
│   └── models/              ← Saved model weights (.zip)
│
├── dashboard/
│   └── app.py               ← Dash live training dashboard
│
├── modes/
│   ├── training_mode.py     ← Watch trained agent play
│   └── race_mode.py         ← Human vs AI race
│
└── training_logs/           ← CSV logs (auto-created during training)
```

---

## Algorithm Comparison

| Algorithm | Style | Strengths | Expected Steps |
|---|---|---|---|
| DQN | Off-policy, experience replay | Sample efficient, discrete actions | 300 k |
| PPO | On-policy, clipped objective | Stable, reliable convergence | 500 k |
| A2C | On-policy, advantage actor-critic | Fast, low memory | 400 k |

After training, the dashboard shows a side-by-side comparison of average
reward over the last 100 episodes.

---

## Reward Design

```
+10   collect a gem
+150  reach the exit door (win)
−100  die (fall into wrong element or acid)
−0.05 per RL step (time penalty — encourages speed)
+0.3  shaping bonus for moving toward nearest gem
```

---

## State Space (35 features)

| Feature | Size |
|---|---|
| Agent (x, y) normalised | 2 |
| Agent velocity (vx, vy) normalised | 2 |
| on_ground flag | 1 |
| Relative direction to nearest gem | 2 |
| Relative direction to exit door | 2 |
| Gems remaining (normalised) | 1 |
| Local 5×5 tile grid around agent | 25 |
| **Total** | **35** |

---

## Map Legend

| Symbol | Meaning |
|---|---|
| `#` | Wall (solid) |
| `.` | Platform (solid) |
| `^` | Fire hazard — safe for FireAgent, kills WaterAgent |
| `~` | Water hazard — safe for WaterAgent, kills FireAgent |
| `@` | Acid — kills both |
| `f` | Fire gem — collected by FireAgent |
| `w` | Water gem — collected by WaterAgent |
| `F` | FireAgent spawn |
| `W` | WaterAgent spawn |
| `D` | FireAgent door (exit) |
| `E` | WaterAgent door (exit) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'stable_baselines3'`**
→ Run `pip install -r requirements.txt` inside the active virtual environment.

**pygame window does not open**
→ Make sure you are running on a machine with a display (not headless SSH).

**Training is slow**
→ Reduce `total_timesteps` in `rl/train.py` `ALGO_CONFIGS` for a quick test run.

**`FileNotFoundError: No saved model`**
→ Train first: `python rl/train.py --algo dqn`
