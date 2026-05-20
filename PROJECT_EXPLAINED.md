# Element Quest — Complete Project Explanation

A deep-dive into every file, every design decision, and the full flow of a Fireboy-and-Watergirl-inspired 2D platformer where reinforcement learning agents learn to play from scratch.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Project Structure](#2-project-structure)
3. [Dependencies](#3-dependencies)
4. [Configuration — config.py](#4-configuration--configpy)
5. [The Game Layer](#5-the-game-layer)
   - [tiles.py — Constants and Rendering](#tilespyconstants-and-rendering)
   - [maps/ — Level Design](#maps--level-design)
   - [characters.py — Physics and Characters](#characterspy--physics-and-characters)
   - [engine.py — Core Game Loop](#enginepy--core-game-loop)
6. [The RL Layer](#6-the-rl-layer)
   - [env.py — Gymnasium Environment](#envpy--gymnasium-environment)
   - [train.py — Training Pipeline](#trainpy--training-pipeline)
7. [Modes](#7-modes)
   - [training_mode.py — Watch Mode](#training_modepy--watch-mode)
   - [race_mode.py — Human vs AI](#race_modepy--human-vs-ai)
8. [Dashboard — app.py](#8-dashboard--apppy)
9. [Main Menu — main.py](#9-main-menu--mainpy)
10. [Full Data Flow](#10-full-data-flow)
11. [RL Algorithm Details](#11-rl-algorithm-details)
12. [Reward Engineering](#12-reward-engineering)
13. [Observation Space (35 Features)](#13-observation-space-35-features)
14. [How to Run](#14-how-to-run)

---

## 1. What Is This Project?

**Element Quest** is a complete reinforcement learning research and demo project. At its core:

- There is a 2D grid-based platformer game (similar to Fireboy and Watergirl).
- Two characters exist: **FireAgent** (red) and **WaterAgent** (blue).
- Each character has its own hazards, gems, and exit door.
- Three RL algorithms — **DQN**, **PPO**, and **A2C** — are trained to control FireAgent.
- After training, you can **watch** the AI play, or **race** against it yourself as WaterAgent.
- A **live web dashboard** lets you monitor training progress in real time.

The project is entirely CPU-trainable, requires no GPU, and runs on Windows/Mac/Linux.

---

## 2. Project Structure

```
Multi-Agent-Game-Using-RL/
│
├── main.py                    ← Entry point: animated main menu
├── config.py                  ← Training render settings
├── requirements.txt           ← All Python dependencies
│
├── game/                      ← Pure game logic (no RL here)
│   ├── tiles.py               ← Tile types, colors, constants, draw functions
│   ├── characters.py          ← FireAgent & WaterAgent with full physics
│   ├── engine.py              ← Manages game state, step, render
│   └── maps/
│       ├── map_training.py    ← 20×13 single-agent training level
│       └── map_race.py        ← 20×13 two-agent race level
│
├── rl/                        ← Reinforcement learning logic
│   ├── env.py                 ← Gymnasium wrapper around GameEngine
│   ├── train.py               ← DQN/PPO/A2C training, CSV logging, callbacks
│   └── models/                ← Saved .zip model weights after training
│
├── dashboard/
│   └── app.py                 ← Dash web app: live reward/length charts
│
├── modes/
│   ├── training_mode.py       ← Watch a trained agent play
│   └── race_mode.py           ← Human vs AI race
│
└── training_logs/             ← Auto-created CSVs during training
    ├── dqn_log.csv
    ├── ppo_log.csv
    └── a2c_log.csv
```

---

## 3. Dependencies

Defined in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame-ce` | >=2.5.0 | Rendering the game window, handling keyboard input |
| `gymnasium` | >=0.29.0 | Standard RL environment interface (obs/action/step/reset) |
| `stable-baselines3` | >=2.3.0 | DQN, PPO, A2C implementations with MlpPolicy |
| `torch` | >=2.0.0 | Neural network backend for Stable-Baselines3 |
| `numpy` | >=1.24.0 | Observation vector construction, numerical operations |
| `dash` | >=2.14.0 | Web framework for the live dashboard |
| `plotly` | >=5.17.0 | Interactive charts inside Dash |
| `pandas` | >=2.0.0 | Reading CSVs, rolling averages for smooth reward curves |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## 4. Configuration — `config.py`

```python
RENDER_TRAINING      = False   # True = live pygame window during training
RENDER_PREVIEW_EVERY = 0       # Preview window every N episodes (0 = off)
RENDER_FPS           = 300     # FPS cap for training window
```

This file is read at the start of every training run. No restart needed — just edit and re-run.

- **RENDER_TRAINING = True**: A pygame window opens showing the agent playing every episode. Slower but lets you watch progress.
- **RENDER_TRAINING = False** (default): Headless training — fastest. Optionally shows a short preview window every N episodes.
- **RENDER_FPS**: Caps the frame rate if rendering. 300 is fast enough to watch, 0 means uncapped.

---

## 5. The Game Layer

### `tiles.py` — Constants and Rendering

This file defines everything about the visual and logical structure of the world.

**Grid constants:**
```python
TILE_SIZE  = 48    # each tile is 48×48 pixels
GRID_COLS  = 20    # map is 20 tiles wide
GRID_ROWS  = 13    # map is 13 tiles tall
CHAR_WIDTH  = 32   # character sprite width
CHAR_HEIGHT = 40   # character sprite height
UI_HEIGHT   = 70   # top bar for score/status display
WINDOW_W    = 960  # total window width  (20 × 48)
WINDOW_H    = 694  # total window height (13 × 48 + 70)
```

**Tile enum (integer encoding):**
```python
class Tile(IntEnum):
    EMPTY       = 0   # open air
    PLATFORM    = 1   # walkable solid surface
    FIRE_HAZARD = 2   # kills WaterAgent, safe for FireAgent
    WATER_HAZARD= 3   # kills FireAgent, safe for WaterAgent
    ACID        = 4   # kills both agents
    FIRE_GEM    = 5   # collectible for FireAgent only
    WATER_GEM   = 6   # collectible for WaterAgent only
    FIRE_DOOR   = 7   # exit for FireAgent
    WATER_DOOR  = 8   # exit for WaterAgent
    WALL        = 9   # solid, impassable
```

The integer encoding matters: the grid is stored as a 2D list of ints, and these ints are passed directly into the observation vector (divided by 9.0 to normalize to [0, 1]).

**Collision sets:**
```python
SOLID_TILES      = {Tile.WALL, Tile.PLATFORM}
LETHAL_TO_FIRE   = {Tile.WATER_HAZARD, Tile.ACID}
LETHAL_TO_WATER  = {Tile.FIRE_HAZARD, Tile.ACID}
```

**`draw_tile(surface, tile, rect, frame)`**: Renders each tile type with animated effects using `frame` as the animation counter:
- Fire hazard: flickering flame polygons
- Water hazard: animated wave stripe
- Acid: rising bubble particle
- Gems: glowing diamond shape with pulsing brightness
- Doors: colored archway with a knob detail

---

### `maps/` — Level Design

Maps are defined as plain ASCII strings, then parsed into 2D integer grids.

**`map_training.py`** — single FireAgent, 20×13:
```
####################
#                  #
#F  f              #
####   ..          #
#        f    ...  #
#  ..  ......      #
#        ~~~   f   #
#  ....        ... #
#      f  ..       #
#  .. ....  ~~~    #
#         f    ..  #
#  f  ..        f  D
####################
```
- `F` = FireAgent spawn at column 1, row 2
- `f` = Fire gems scattered across platforms
- `~` = Water hazard (kills FireAgent — the agent must learn to avoid these)
- `D` = Fire door (exit) at top-right
- No `W`, `w`, or `E` — only FireAgent is used in training

**`map_race.py`** — both agents on same map, with separate hazards, gems, and exits.

**`load()` function:**
```python
def load():
    # Returns: (grid, fire_spawn, water_spawn)
    # grid is a list[list[int]] of Tile values
```
Parses each character using `_CHAR_MAP`, pads/trims rows to exactly 20 columns, and extracts spawn positions dynamically from `F`/`W` characters.

---

### `characters.py` — Physics and Characters

**Physics constants:**
```python
GRAVITY    = 0.7    # added to vy every frame
JUMP_VEL   = -14.0  # initial upward velocity on jump
MOVE_SPEED = 4.0    # pixels per frame left/right
MAX_FALL   = 16.0   # terminal velocity downward
```

**Action space:**
```python
ACTION_NOOP  = 0   # do nothing (vx = 0)
ACTION_LEFT  = 1   # vx = -4.0, facing = -1
ACTION_RIGHT = 2   # vx = +4.0, facing = +1
ACTION_JUMP  = 3   # vy = -14.0 only if on_ground
```

**`Character` class:**

Each character has:
- `px`, `py` — pixel position
- `vx`, `vy` — velocity
- `on_ground` — boolean, only jump when True
- `alive`, `at_door` — terminal states
- `gems` — count of collected gems
- `facing` — direction for sprite rendering (1=right, -1=left)

**`apply_action(action)`**: Sets velocity based on the action. Jump only works when on the ground. Noop sets `vx = 0` (no sliding).

**`update(grid)`**: One physics frame in two steps:
1. Apply gravity: `vy = min(vy + 0.7, 16.0)`
2. Move horizontally + resolve wall collisions
3. Move vertically + resolve floor/ceiling collisions
4. Check tile interactions (gems, hazards, door, out-of-bounds)

**Collision resolution** works by checking the character's corners against solid tiles:
- Horizontal: checks front edge at top and bottom of character
- Vertical: checks bottom edge at left and right of character (landing), top edge (ceiling)

**`_check_tiles(grid, events)`**: Detects the tile under the center and foot of the character:
- FireAgent on `FIRE_GEM` → `events['collect_gem']`, `gems += 1`
- FireAgent on `WATER_HAZARD` or `ACID` → `alive = False`, `events['died']`
- FireAgent on `FIRE_DOOR` → `at_door = True`, `events['reached_door']`
- Out of bounds → `alive = False`

**Rendering**: Characters are drawn purely with `pygame.draw` calls — no image files needed:
- FireAgent: orange body, circular head, flame crown that flickers with `frame`, animated legs
- WaterAgent: blue body, water droplet crown that bobs up/down, same leg animation

---

### `engine.py` — Core Game Loop

`GameEngine` is the single source of truth for game state.

**Constructor:**
```python
GameEngine(map_module, agents='fire', render_mode=None)
```
- `agents='fire'` → only FireAgent active (for RL training)
- `agents='both'` → both characters active (for race mode)
- `render_mode='human'` → initializes pygame window

**`reset()`**: Deep-copies the base grid, resets frame counter, creates fresh `Character` instances at their spawn positions.

**`step(fire_action, water_action)`**:
```python
def step(self, fire_action=0, water_action=0) -> dict:
```
- Calls `apply_action()` then `update()` for each active, alive character
- Removes collected gems from the grid (`grid[row][col] = Tile.EMPTY`)
- Returns `{'fire': {...events...}, 'water': {...events...}}`
- Increments `self.frame`

**`render()`**: Fills background, draws UI bar (gems, status, frame counter), draws all non-empty tiles, draws characters. Calls `pygame.display.flip()`.

**`gem_positions(gem_tile)`**: Scans the grid and returns all `(col, row)` positions of a specific tile type. Used by the RL environment to compute distances for reward shaping and observations.

---

## 6. The RL Layer

### `env.py` — Gymnasium Environment

`ElementQuestEnv` wraps `GameEngine` in the standard Gymnasium interface that Stable-Baselines3 expects.

**Interface:**
```
observation_space: Box(float32, shape=(35,), low=-1.0, high=1.0)
action_space:      Discrete(4)
```

**Constructor parameters:**
- `map_module` — which map to use (defaults to `map_training`)
- `render_mode` — `'human'` or `None`
- `frame_skip=4` — each RL action is repeated for 4 physics frames
- `render_fps=60` — FPS cap during rendering

**`frame_skip`** is important: the agent makes one decision, but the game runs 4 physics steps. This means:
- The agent doesn't need to act at 60fps
- Actions have more physical effect (jump goes higher, movement covers more ground)
- Training is more efficient (fewer decisions needed)

**`reset()`**: Calls `engine.reset()`, clears step counter and previous gem distance, returns initial observation.

**`step(action)`**:
```python
def step(self, action: int):
    reward = 0.0
    for _ in range(frame_skip):        # repeat action 4 times
        events = engine.step(action)
        if 'collect_gem' in events:    reward += R_GEM    # +10
        if 'reached_door' in events:   reward += R_WIN    # +150, terminate
        if 'died' in events:           reward += R_DEATH  # -100, terminate
    reward += R_TIME                   # -0.05 per step
    # reward shaping: +0.3 if closer to nearest gem
    if step_count >= MAX_STEPS:
        truncated = True               # 1200 step cap
```

**`_get_obs()`** — constructs the 35-feature vector:
```python
[
    norm_x,          # px / (20 * 48)         → position X
    norm_y,          # py / (13 * 48)         → position Y
    norm_vx,         # clip(vx / 10, -1, 1)   → velocity X
    norm_vy,         # clip(vy / 16, -1, 1)   → velocity Y
    on_ground,       # 0.0 or 1.0
    gem_dx,          # direction to nearest fire gem X
    gem_dy,          # direction to nearest fire gem Y
    door_dx,         # direction to fire door X
    door_dy,         # direction to fire door Y
    gems_remaining,  # count / total (how many left)
    *local_grid,     # 5×5 = 25 tiles around agent, each / 9.0
]
```

The 5×5 local grid scans `row ± 2, col ± 2` around the agent's center tile. Out-of-bounds cells are filled with `1.0` (treated as wall).

**`_handle_window_events()`**: If the user closes the pygame window during training, the environment silently switches to headless mode instead of crashing.

---

### `train.py` — Training Pipeline

**Hyperparameter configs (`ALGO_CONFIGS`):**

```python
"dqn": dict(
    cls=DQN,
    total_timesteps=300_000,
    kwargs=dict(
        policy="MlpPolicy",          # fully-connected neural net
        learning_rate=1e-3,
        buffer_size=50_000,          # replay buffer size
        learning_starts=2_000,       # steps before first update
        batch_size=64,
        gamma=0.99,                  # discount factor
        exploration_fraction=0.3,    # epsilon decays over 30% of training
        exploration_final_eps=0.05,  # final epsilon = 5%
        train_freq=4,                # update every 4 steps
        target_update_interval=500,  # sync target network every 500 steps
    )
)

"ppo": dict(
    cls=PPO,
    total_timesteps=500_000,
    kwargs=dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=2048,        # steps per rollout buffer
        batch_size=64,
        n_epochs=10,         # gradient updates per rollout
        gamma=0.99,
        gae_lambda=0.95,     # GAE smoothing
        clip_range=0.2,      # PPO clipping threshold
    )
)

"a2c": dict(
    cls=A2C,
    total_timesteps=400_000,
    kwargs=dict(
        policy="MlpPolicy",
        learning_rate=7e-4,
        n_steps=5,           # much shorter rollout than PPO
        gamma=0.99,
        gae_lambda=1.0,      # no GAE smoothing (full returns)
        ent_coef=0.01,       # entropy bonus for exploration
    )
)
```

**`TrainingLogger` callback:**

Extends `BaseCallback` from Stable-Baselines3. Called after every step:
- Accumulates episode reward
- On episode end: writes one row to `training_logs/<algo>_log.csv`
  - columns: `episode, timestep, reward, ep_length, elapsed_s, algo`
- Prints a summary every 50 episodes

The CSV is written in append mode so the dashboard can read it live while training runs.

**`LiveRenderCallback` callback:**

Only active when `RENDER_TRAINING = False` and `RENDER_PREVIEW_EVERY > 0`. Every N episodes:
1. Creates a temporary `ElementQuestEnv(render_mode='human')`
2. Runs one full episode using `model.predict(obs, deterministic=True)`
3. Shows the agent playing — closes when episode ends or user presses ESC
4. Destroys the window and resumes headless training

**`train_algo(algo_name)`** — main training function:
```
1. Read config (render mode, FPS, preview interval)
2. Create environment with Monitor wrapper (for episode stats)
3. Instantiate model (DQN/PPO/A2C) with hyperparameters
4. Call model.learn(total_timesteps, callbacks=[...])
5. Save model to rl/models/<algo>_model.zip
```

**`load_model(algo_name)`**: Loads a `.zip` model. Raises `FileNotFoundError` if not trained yet.

**`reset_all()`**: Deletes all `.csv` files in `training_logs/` and all `.zip` files in `rl/models/`.

---

## 7. Modes

### `training_mode.py` — Watch Mode

Loads a saved model and runs it continuously in the training map.

```python
def run(algo_name='dqn'):
    model = load_model(algo_name)
    env   = ElementQuestEnv(render_mode='human')
    obs, _ = env.reset()

    while running:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode {n} | reward {total:.1f} | gems {info['gems']}")
            obs, _ = env.reset()
```

Key point: `deterministic=True` means the model picks the best action (no exploration noise). You're watching the agent's "best behaviour", not training.

---

### `race_mode.py` — Human vs AI

The most complex mode. Two characters share the same map simultaneously.

**Setup:**
- AI controls **FireAgent** using the loaded trained model
- Human controls **WaterAgent** using keyboard (arrow keys / WASD)
- `GameEngine(agents='both')` activates both characters

**Game loop:**
```python
while running:
    human_action = _keyboard_action()         # read keyboard

    if frame_count % FRAME_SKIP == 0:         # AI acts every 2 frames
        obs = _fire_obs(engine)               # build same 35-feature obs
        ai_action, _ = ai_model.predict(obs, deterministic=True)

    engine.step(fire_action=ai_action, water_action=human_action)
    engine.render()
    check_win_conditions()
```

`FRAME_SKIP = 2` for the AI means it acts every 2 rendered frames, giving the human a slight reflex advantage.

**`_fire_obs(engine)`**: Reconstructs the exact same 35-feature observation the model was trained on. This must match `env.py`'s `_get_obs()` exactly — if they drift, the model will behave erratically.

**Win condition logic (`_resolve_result`)**:
- Reaching your door > everything else
- If both reach door: whoever has more score wins (gems × 10 + 50 door bonus)
- If neither reaches door: death/score comparison
- If time runs out (3000 frames = ~50 seconds): score comparison

**Controls:**
```
Arrow keys / WASD    → move WaterAgent
W / Space            → jump
R                    → restart (only after round ends)
ESC                  → quit
```

---

## 8. Dashboard — `app.py`

A **Dash** web application that reads the CSV log files and renders live charts.

**URL:** `http://127.0.0.1:8050` — run in a separate terminal while training.

**Layout:**
1. **Header bar** — gradient title
2. **Status cards** — one card per algorithm showing: episodes trained, timesteps, last smoothed reward, best reward, status (Training / Complete / Waiting)
3. **Reward chart** — episode reward smoothed over 20 episodes, one line per algorithm
4. **Episode length chart** — how many steps each episode took (shorter = faster = better)
5. **Comparison bar chart** — average reward over last 100 episodes per algorithm

**Auto-refresh:** A `dcc.Interval` component fires every 2 seconds, triggering the `update()` callback which re-reads all CSV files.

**`load_logs()`**:
```python
for algo in ['dqn', 'ppo', 'a2c']:
    df = pd.read_csv(f"training_logs/{algo}_log.csv")
    df['reward_smooth'] = df['reward'].rolling(20, min_periods=1).mean()
```

**Reset button**: The "Clear All Logs & Models" button calls `reset_logs()` which deletes all `.csv` and `.zip` files — same as `python rl/train.py --reset`.

**Color scheme:**
```python
COLORS = {"dqn": "#FF6B35", "ppo": "#4ECDC4", "a2c": "#A855F7"}
# orange        teal          purple
```

---

## 9. Main Menu — `main.py`

The entry point. Renders an animated pygame menu and routes to each mode.

**Animated background (`_draw_bg`)**:
- 5 concentric circles with slowly varying centers using `sin/cos` of the frame counter
- Creates a subtle pulsing orbital effect

**Menu items:**
```python
MENU_ITEMS = [
    ("Train Agents  (DQN / PPO / A2C)", "train"),
    ("Watch Trained Agent Play",         "watch"),
    ("Human vs AI  Race Mode",           "race"),
    ("Open Training Dashboard",          "dashboard"),
    ("Quit",                             "quit"),
]
```

**Navigation:** UP/DOWN arrows cycle selection, ENTER activates, ESC quits.

**Selected item rendering:** Highlighted with a filled rounded rectangle behind it plus a border.

**Algorithm picker (`_pick_algo`)**: A secondary screen that appears after selecting Train/Watch/Race, letting the user pick DQN, PPO, A2C, or Back.

**Mode routing:**
- `train` → quits pygame, runs `train_algo(algo)`, then re-inits pygame menu
- `watch` → quits pygame, runs `modes.training_mode.run(algo)`, re-inits
- `race`  → quits pygame, runs `modes.race_mode.run(algo)`, re-inits
- `dashboard` → launches `dashboard/app.py` as a subprocess, menu stays open

The pattern of `pygame.quit()` → run mode → `_reinit_pygame()` is used because training and race modes each manage their own pygame context.

---

## 10. Full Data Flow

Here is the complete flow from startup to a trained agent racing a human:

```
python main.py
│
├─ pygame menu renders
│
├─ User selects "Train Agents" → picks "DQN"
│   │
│   ├─ pygame.quit()
│   ├─ train_algo("dqn")
│   │   ├─ ElementQuestEnv() created
│   │   │   └─ GameEngine(map_training, agents='fire', render_mode=None)
│   │   │       └─ map_training.load() → grid[13][20], fire_spawn
│   │   │
│   │   ├─ Monitor(env) wraps env for episode stats
│   │   ├─ DQN(env, MlpPolicy, lr=1e-3, ...) created
│   │   │
│   │   ├─ model.learn(300_000 steps, callbacks=[TrainingLogger])
│   │   │   │
│   │   │   └─ For each step:
│   │   │       ├─ env.step(action)
│   │   │       │   ├─ frame_skip × engine.step(action)
│   │   │       │   │   ├─ character.apply_action()
│   │   │       │   │   └─ character.update(grid)  → events
│   │   │       │   ├─ compute reward from events
│   │   │       │   └─ _get_obs() → float32[35]
│   │   │       │
│   │   │       └─ TrainingLogger._on_step()
│   │   │           └─ on episode end: append row to dqn_log.csv
│   │   │
│   │   └─ model.save("rl/models/dqn_model.zip")
│   │
│   └─ _reinit_pygame() → menu reappears
│
├─ User selects "Human vs AI Race Mode" → picks "DQN"
│   │
│   ├─ pygame.quit()
│   ├─ race_mode.run("dqn")
│   │   ├─ load_model("dqn") → DQN.load("rl/models/dqn_model.zip")
│   │   ├─ GameEngine(map_race, agents='both', render_mode='human')
│   │   │
│   │   └─ Game loop at 60fps:
│   │       ├─ keyboard → human_action (WaterAgent)
│   │       ├─ _fire_obs(engine) → float32[35]
│   │       ├─ ai_model.predict(obs) → ai_action (FireAgent)
│   │       ├─ engine.step(fire_action, water_action)
│   │       ├─ engine.render()
│   │       └─ check win conditions → show overlay
│   │
│   └─ _reinit_pygame() → menu reappears
│
└─ User selects "Open Training Dashboard"
    └─ subprocess.Popen(["python", "dashboard/app.py"])
        └─ Dash server at http://127.0.0.1:8050
            └─ Every 2s: read CSVs → update reward/length/comparison charts
```

---

## 11. RL Algorithm Details

### DQN (Deep Q-Network)
- **Type:** Off-policy
- **Core idea:** Learn a Q-function `Q(s, a)` = expected total reward from state `s` taking action `a`. Pick action with highest Q-value.
- **Experience replay:** Stores `(s, a, r, s', done)` tuples in a buffer (50,000 capacity). Trains on random mini-batches to break temporal correlation.
- **Target network:** A frozen copy of the network updated every 500 steps. Prevents unstable training from chasing a moving target.
- **Exploration:** Epsilon-greedy — decays from 1.0 to 0.05 over 30% of training (90,000 steps). After that, mostly greedy.
- **When it shines:** Discrete actions, sample-efficient when the replay buffer is used well.

### PPO (Proximal Policy Optimization)
- **Type:** On-policy, actor-critic
- **Core idea:** Directly learn a policy `π(a|s)`. Clipping prevents the policy from changing too much in one update.
- **Rollout:** Collects 2048 steps of experience, then does 10 gradient update passes on that batch.
- **GAE (λ=0.95):** Generalized Advantage Estimation — smooths the advantage signal between TD(1) and full Monte Carlo returns.
- **Clip range 0.2:** Policy ratio `π_new/π_old` is clipped to `[0.8, 1.2]`. If the ratio goes outside this, the gradient is zeroed.
- **When it shines:** Reliable and stable. Often the best default choice.

### A2C (Advantage Actor-Critic)
- **Type:** On-policy, actor-critic
- **Core idea:** Like PPO but simpler — no clipping, no replay, updates after just 5 steps.
- **Very short rollouts (n_steps=5):** Updates happen extremely frequently. Fast per-update but noisier.
- **Entropy bonus (0.01):** Adds a small reward for policy entropy to maintain exploration.
- **When it shines:** Low memory footprint, fast wall-clock time per update. Noisier than PPO but converges well with enough timesteps.

### Comparison Summary

| Property | DQN | PPO | A2C |
|----------|-----|-----|-----|
| Policy type | Off-policy | On-policy | On-policy |
| Replay buffer | Yes (50k) | No | No |
| Update frequency | Every 4 steps | Every 2048 steps | Every 5 steps |
| Total training steps | 300k | 500k | 400k |
| Exploration method | Epsilon-greedy | Policy entropy | Entropy bonus |
| Stability | Medium | High | Lower |

---

## 12. Reward Engineering

Every reward signal is defined as a constant in `rl/env.py`:

```python
R_GEM      = +10.0    # collecting a fire gem
R_WIN      = +150.0   # reaching the exit door
R_DEATH    = -100.0   # dying (wrong hazard or falling)
R_TIME     = -0.05    # applied every RL step (not every physics frame)
R_PROGRESS = +0.3     # shaping bonus per tile moved toward nearest gem
```

**Why these values?**

- `R_WIN >> R_GEM`: The agent is incentivized to win even if it misses gems. Gems are secondary.
- `R_DEATH` is large and negative: dying is the worst outcome, worse than losing all gems.
- `R_TIME` is small: discourages standing still without being so large it rushes the agent into hazards.
- `R_PROGRESS` is a shaping reward: guides early training by rewarding movement toward gems, even before the agent can reliably reach them. This prevents early-episode wandering.

**Shaping reward calculation:**
```python
dists = [manhattan_distance(agent, gem) for gem in gem_positions]
nearest = min(dists)
if nearest < prev_nearest:
    reward += R_PROGRESS * ((prev_nearest - nearest) / TILE_SIZE)
```

The improvement is normalized by `TILE_SIZE` (48px) so the bonus scales with tiles moved, not raw pixels.

---

## 13. Observation Space (35 Features)

The agent never sees the full screen or pixels. It sees a compact 35-element float32 vector:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `norm_x` | [0, 1] | Agent X position / map width |
| 1 | `norm_y` | [0, 1] | Agent Y position / map height |
| 2 | `norm_vx` | [-1, 1] | Horizontal velocity / 10 |
| 3 | `norm_vy` | [-1, 1] | Vertical velocity / 16 |
| 4 | `on_ground` | {0, 1} | Is the agent touching the floor |
| 5 | `gem_dx` | [-1, 1] | X direction to nearest fire gem |
| 6 | `gem_dy` | [-1, 1] | Y direction to nearest fire gem |
| 7 | `door_dx` | [-1, 1] | X direction to fire door |
| 8 | `door_dy` | [-1, 1] | Y direction to fire door |
| 9 | `gems_remaining` | [0, 1] | Gems left / total gems |
| 10–34 | `local_grid` | [0, 1] | 5×5 tile grid around agent, each tile / 9.0 |

**Why feature vectors instead of pixels?**
- Much smaller input (35 vs 960×624×3 ≈ 1.8M pixels)
- Trains on CPU in minutes/hours instead of days on GPU
- The policy net is a simple MLP (3 hidden layers), not a CNN
- The agent has exactly the information it needs — position, velocity, nearby tiles, gem/door directions

**Local 5×5 grid:**

The agent's center tile is at the middle of the 5×5 window. Each tile value is divided by 9.0 (max tile integer) to normalize to [0, 1]. Out-of-bounds positions are filled with 1.0 (treated as wall) to prevent index errors near map edges.

---

## 14. How to Run

### Setup
```bash
cd Multi-Agent-Game-Using-RL
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Launch main menu
```bash
python main.py
```
Use arrow keys to navigate, Enter to select.

### Train from command line
```bash
python rl/train.py --algo dqn          # train DQN (300k steps)
python rl/train.py --algo ppo          # train PPO (500k steps)
python rl/train.py --algo a2c          # train A2C (400k steps)
python rl/train.py --algo all          # all three sequentially
python rl/train.py --reset             # wipe all logs and models
```

### Open live dashboard (run while training)
```bash
# In a second terminal:
python dashboard/app.py
# Then open: http://127.0.0.1:8050
```

### Watch trained agent
```bash
python -c "from modes.training_mode import run; run('dqn')"
```

### Race against AI
```bash
python -c "from modes.race_mode import run; run('dqn')"
```
Controls: Arrow keys / WASD to move, W or Space to jump, R to restart, ESC to quit.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: stable_baselines3` | `pip install -r requirements.txt` inside your venv |
| `FileNotFoundError: No saved model` | Train first: `python rl/train.py --algo dqn` |
| Pygame window doesn't open | You need a display — can't run via headless SSH |
| Training is very slow | Set `RENDER_TRAINING = False` in `config.py`, or reduce `total_timesteps` in `train.py` |
| Dashboard shows nothing | Start training first — dashboard reads CSVs that only exist after training begins |

---

*Project by SajeelTariq. Documented in full by Claude.*
