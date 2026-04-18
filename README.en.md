# RL Lab

An interactive laboratory for experimenting with reinforcement learning algorithms. Currently features a Q-learning agent learning to balance an inverted pendulum with real-time visualization.

![Training mode showing pendulum and Q-table heatmap](assets/q-learning-pendulum.mov)

## Overview

RL Lab is an environment for experimenting with reinforcement learning algorithms. The current environment trains a tabular Q-learning agent to balance an inverted pendulum using discrete state and action spaces. The simulation runs in a Pygame window with two panels: the pendulum physics on the left, and a live Q-value heatmap on the right. You can toggle between training and watching the learned policy at any time.

The project is built around a generic simulation loop and a `Simulation` abstract base class, so new environments can be plugged in without touching the training logic.

## Features

- **Tabular Q-learning** with epsilon-greedy exploration and configurable hyperparameters
- **Real-time Q-table visualization** — 2D heatmap of state × action values plus an action frequency histogram
- **Flexible state space** — full angle + angular velocity, or angle-only via `--no-speed`
- **Two reward modes** — simple upright reward, or a shaped reward that penalizes jitter (`--better-reward`)
- **Modular design** — generic game loop works with any class implementing the `Simulation` ABC
- **Interactive physics demo** — test the pendulum manually with keyboard controls

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd rl_lab
uv sync
```

## Usage

### Train the agent

```bash
uv run python -m rl_lab.main
```

| Flag | Description |
|---|---|
| `--no-speed` | Ignore angular velocity; state = angle only |
| `--better-reward` | Shaped reward with gentleness bonuses (less jitter) |

### Interactive controls

| Key | Action |
|---|---|
| `Space` | Toggle training / watch mode |
| `R` | Reset current episode |
| `Q` / `Esc` | Quit |

### Physics demo

Run the physics engine standalone and control the pendulum manually:

```bash
uv run python -m rl_lab.pendulum.physics
```

Arrow keys apply torque; `R` resets; `Q` quits.

## How It Works

### Reinforcement Learning

Reinforcement learning (RL) is about learning what to do — how to map situations to actions — so as to maximize a numerical reward signal. Unlike supervised learning, the agent is not told which actions to take: it must discover which actions yield the most reward by trying them out. Its decisions may affect not only the immediate reward but also the next state and all subsequent rewards.

This introduces the fundamental dilemma of RL: the trade-off between **exploration** (trying new actions to discover better strategies) and **exploitation** (choosing actions already known to work). Neither can be pursued exclusively without failing at the task.

> *"Reinforcement learning problems involve learning what to do — how to map situations to actions — so as to maximize a numerical reward signal."* — Sutton & Barto, [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

### Q-Learning

A model-free, off-policy algorithm that learns through trial-and-error without needing a model of the environment. It maintains a Q-table with the expected value for each (state, action) pair, updated via the Bellman equation:

```
Q[s, a] += α · (r + γ · max(Q[s']) − Q[s, a])
```

`α` controls how much weight new information gets; `γ` sets the importance of future rewards. Exploration is handled with ε-greedy: the agent picks random actions with probability ε, which decays as it learns.

→ [Detailed Q-learning explanation](docs/q-learning.en.md)

Default hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate α | 0.15 |
| Discount γ | 0.99 |
| Epsilon decay | 0.995 |

### State & Action Space

The continuous pendulum state (angle, angular velocity) is discretized into a 32 × 16 grid. The action space consists of 20 evenly spaced torques in [−40, +40] N·m.

### Pendulum Physics

The physics engine integrates the rigid-body equations of motion using Euler integration with gravity (9.81 m/s²), damping, and applied torque. An episode ends when the rod tension exceeds 200 N (i.e., the pendulum swings too hard).

## Project Structure

```
src/rl_lab/
├── main.py                  # Generic Q-learning game loop
├── agents/
│   └── q_agent.py           # Tabular Q-learning agent
├── engine/
│   ├── simulation.py        # Simulation abstract base class
│   └── policy_renderer.py   # Q-value heatmap renderer
└── pendulum/
    ├── physics.py           # Pendulum physics engine
    ├── env.py               # Gym-style environment wrapper
    └── sim.py               # PendulumSim concrete implementation
```

## Extending

To add a new environment, implement the `Simulation` ABC in `src/rl_lab/engine/simulation.py` and pass an instance to the game loop in `main.py`. The Q-learning agent and policy renderer are environment-agnostic.

## Development

```bash
uv sync
pre-commit install
```

Code quality is enforced with Ruff (linting + formatting) and Pyright (type checking).
