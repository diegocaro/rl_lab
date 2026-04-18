"""
Q-Learning Pendulum Balancer
============================
Goal: Keep a pendulum upright (θ ≈ 0) using Q-learning.

Physics:
  θ̈ = (g/L)·sin(θ) - (b/mL²)·θ̇ + (1/mL²)·τ

State space:  discretized (angle, angular velocity)
Action space: torque ∈ {-τ_max, …, +τ_max}
Reward:       +1 each step the pendulum is "up" (|θ| < 0.2 rad ≈ 11°)

Controls:
  SPACE  – toggle training / watching
  R      – reset episode
  Q      – quit
"""

import argparse
import math
import sys
import time

import numpy as np
import pygame

from pendulum import MAX_TORQUE, PendulumRenderer
from pendulum_env import PendulumEnv
from policy_renderer import PolicyRenderer
from q_agent import QLearningAgent

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Q-Learning Pendulum Balancer")
_parser.add_argument(
    "--no-speed",
    action="store_true",
    help="Ignore angular velocity in the state (angle-only Q-table).",
)
_reward_group = _parser.add_mutually_exclusive_group()
_reward_group.add_argument(
    "--simple-reward",
    action="store_true",
    help="Reward: +1 when upright, -1 on termination (default).",
)
_reward_group.add_argument(
    "--better-reward",
    action="store_true",
    help="Reward: upright bonus minus velocity and torque penalties to reduce jitter.",
)
_args = _parser.parse_args()
USE_SPEED = not _args.no_speed
USE_BETTER_REWARD = _args.better_reward

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_ANGLE = 32
N_SPEED = 16
MAX_SPEED = 20.0
ACTIONS = np.linspace(-MAX_TORQUE, MAX_TORQUE, 20)

MAX_STEPS = 500

# ── Pendulum state discretizer ────────────────────────────────────────────────
_angle_bins = np.linspace(-math.pi, math.pi, N_ANGLE + 1)
_speed_bins = np.linspace(-MAX_SPEED, MAX_SPEED, N_SPEED + 1)


def _make_discretize(use_speed: bool):
    def discretize(obs):
        theta, theta_dot = obs
        a = int(np.clip(np.digitize(theta, _angle_bins) - 1, 0, N_ANGLE - 1))
        if use_speed:
            v = int(np.clip(np.digitize(theta_dot, _speed_bins) - 1, 0, N_SPEED - 1))
            return (a, v)
        return (a,)
    return discretize

# ── Pygame setup ──────────────────────────────────────────────────────────────
W_PEND, W_HEAT, H = 600, 400, 550
W = W_PEND + W_HEAT
CX, CY = W_PEND // 2, 260
PX_LEN = 180
FPS_WATCH = 60

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Q-Learning Pendulum")
clock = pygame.time.Clock()

pend_renderer = PendulumRenderer(width=W, height=H, cx=CX, cy=CY, scale=PX_LEN, show_hud=True)

policy_renderer = PolicyRenderer(
    n_states=N_ANGLE,
    n_actions=len(ACTIONS),
    action_range=(-MAX_TORQUE, MAX_TORQUE),
    action_name="torque",
    state_ticks=[
        (0.0,  "+pi  "),
        (0.25, "+pi/2"),
        (0.5,  "  0  "),
        (0.75, "-pi/2"),
        (1.0,  "-pi  "),
    ],
)

# Fonts & colours
font_big = pygame.font.SysFont("monospace", 22, bold=True)
font_med = pygame.font.SysFont("monospace", 16)
font_sml = pygame.font.SysFont("monospace", 13)

BG      = (15, 17, 26)
HINT_C  = (90, 100, 120)
BOB_UP_C = (80, 255, 140)
TORQUE_C = (255, 180, 50)
TEXT_C  = (200, 210, 230)
WARN_C  = (255, 80, 80)
GRID_C  = (30, 35, 50)


# ── Pendulum panel renderer ───────────────────────────────────────────────────
def draw_pendulum(env, last_torque, episode, step, reward_total, epsilon, training, fps_actual):
    pend_renderer.draw(env.theta, env.theta_dot, last_torque, env.theta_ddot)
    screen.blit(pend_renderer.surface, (0, 0))

    upright = abs(env.theta) < 0.2

    for r in range(50, 300, 50):
        pygame.draw.circle(screen, GRID_C, (CX, CY), r, 1)
    for angle_deg in range(0, 360, 30):
        rad = math.radians(angle_deg)
        pygame.draw.line(
            screen, GRID_C, (CX, CY),
            (CX + int(250 * math.cos(rad)), CY + int(250 * math.sin(rad))), 1,
        )

    pygame.draw.arc(
        screen, (40, 80, 40),
        (CX - PX_LEN, CY - PX_LEN, 2 * PX_LEN, 2 * PX_LEN),
        math.pi / 2 - 0.2, math.pi / 2 + 0.2, 4,
    )

    if last_torque != 0:
        arc_r = 30
        start = -math.pi / 2 + (0 if last_torque < 0 else math.pi)
        end = start + math.pi * (last_torque / MAX_TORQUE)
        if start > end:
            start, end = end, start
        pygame.draw.arc(
            screen, TORQUE_C,
            (CX - arc_r, CY - arc_r, 2 * arc_r, 2 * arc_r),
            start, end, 3,
        )

    mode_str = "TRAINING" if training else "WATCHING"
    mode_col = WARN_C if training else BOB_UP_C
    mode_surf = font_big.render(mode_str, True, mode_col)
    screen.blit(mode_surf, (W_PEND - mode_surf.get_width() - 20, H - mode_surf.get_height() - 20))

    lh, y0 = 22, 330

    def txt(label, value, col=TEXT_C):
        nonlocal y0
        screen.blit(font_med.render(f"{label:<14}{value}", True, col), (30, y0))
        y0 += lh

    txt("Episode:", f"{episode}")
    txt("Step:", f"{step}/{MAX_STEPS}")
    txt("State:", "angle+speed" if USE_SPEED else "angle only", col=HINT_C)
    txt("Upright:", f"{reward_total:.0f} steps")
    txt("Epsilon:", f"{epsilon:.3f}")
    txt("FPS:", f"{fps_actual:.0f}")

    screen.blit(
        font_sml.render("SPACE: train/watch   R: reset   Q: quit", True, HINT_C),
        (30, H - 30),
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    env = PendulumEnv(better_reward=USE_BETTER_REWARD)
    q_shape = (N_ANGLE, N_SPEED) if USE_SPEED else (N_ANGLE,)
    agent = QLearningAgent(
        q_shape=q_shape,
        n_actions=len(ACTIONS),
        discretize=_make_discretize(USE_SPEED),
    )

    training = True
    episode = 0
    fps_actual = 0.0

    obs = env.reset()
    step = 0
    reward_total = 0.0
    last_torque = 0.0
    last_act_idx = 0
    torque_counts = np.zeros(agent.n_actions, dtype=int)
    t_last = time.perf_counter()

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    obs = env.reset()
                    step = 0
                    reward_total = 0.0
                    torque_counts[:] = 0
                if event.key == pygame.K_SPACE:
                    training = not training

        # ── Step ──────────────────────────────────────────────────────────────
        act_idx = agent.act(obs, explore=training)
        last_torque = ACTIONS[act_idx]
        last_act_idx = act_idx
        torque_counts[act_idx] += 1

        next_obs, reward, done = env.step(last_torque)
        reward_total += reward
        step += 1

        if training:
            agent.learn(obs, act_idx, reward, next_obs)

        obs = next_obs

        # ── Episode end ───────────────────────────────────────────────────────
        if done or step >= MAX_STEPS:
            if training:
                agent.end_episode()
            episode += 1
            obs = env.reset()
            step = 0
            reward_total = 0.0
            torque_counts[:] = 0

        # ── Render ────────────────────────────────────────────────────────────
        draw_pendulum(env, last_torque, episode, step, reward_total, agent.epsilon, training, fps_actual)
        policy_renderer.draw(
            screen, (W_PEND, 0, W_HEAT, H),
            agent.Q.max(axis=1) if USE_SPEED else agent.Q,
            (math.pi - env.theta) / (2 * math.pi),
            torque_counts, last_act_idx,
            state_label=f"{env.theta:+.2f}",
        )
        pygame.display.flip()

        # ── Timing ────────────────────────────────────────────────────────────
        clock.tick(0 if training else FPS_WATCH)
        now = time.perf_counter()
        fps_actual = 1.0 / max(now - t_last, 1e-6)
        t_last = now


if __name__ == "__main__":
    main()
