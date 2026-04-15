"""
Q-Learning Pendulum Balancer
============================
Goal: Keep a pendulum upright (θ ≈ 0) using Q-learning.

Physics:
  θ̈ = (g/L)·sin(θ) - (b/mL²)·θ̇ + (1/mL²)·τ

State space:  discretized (angle, angular velocity)
Action space: torque ∈ {-τ_max, 0, +τ_max}
Reward:       +1 each step the pendulum is "up" (|θ| < 0.2 rad ≈ 11°)

Controls:
  SPACE  – toggle training / watching
  R      – reset episode
  Q      – quit
"""

import math
import sys
import time

import numpy as np
import pygame

from pendulum import MAX_TORQUE, Pendulum, PendulumRenderer

# ── Q-learning hyper-parameters ────────────────────────────────────────────────
MAX_SPEED = 16.0  # rad/s — discretization bound for the Q-table speed bins
N_ANGLE = 32      # angle bins   (–π … π)
N_SPEED = 32      # speed bins   (–MAX_SPEED … MAX_SPEED)
ACTIONS = np.linspace(-MAX_TORQUE, MAX_TORQUE, 9)  # 9 actions
N_ACTS = len(ACTIONS)

ALPHA = 0.15  # learning rate
GAMMA = 0.99  # discount
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995  # per episode

MAX_STEPS = 500  # steps per episode

# ── Discretisation helpers ─────────────────────────────────────────────────────
angle_bins = np.linspace(-math.pi, math.pi, N_ANGLE + 1)
speed_bins = np.linspace(-MAX_SPEED, MAX_SPEED, N_SPEED + 1)


def discretise(theta, theta_dot):
    a = np.clip(np.digitize(theta, angle_bins) - 1, 0, N_ANGLE - 1)
    s = np.clip(np.digitize(theta_dot, speed_bins) - 1, 0, N_SPEED - 1)
    return a, s


# ── Q-table ────────────────────────────────────────────────────────────────────
Q = np.zeros((N_ANGLE, N_SPEED, N_ACTS))


def choose_action(a, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTS)
    return int(np.argmax(Q[a, s]))


def update_q(a, s, act, reward, a2, s2):
    best_next = np.max(Q[a2, s2])
    Q[a, s, act] += ALPHA * (reward + GAMMA * best_next - Q[a, s, act])


# ── Pygame visualisation ───────────────────────────────────────────────────────
W, H = 600, 550
CX, CY = W // 2, 260  # pivot centre
PX_LEN = 180  # pixels per metre
PIVOT_R = 8
BOB_R = 18
FPS_TRAIN = 0  # uncapped during training
FPS_WATCH = 60

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Q-Learning Pendulum")
clock = pygame.time.Clock()

# Pivot is horizontally centred but raised to leave room for the HUD below
renderer = PendulumRenderer(width=W, height=H, cx=CX, cy=CY, scale=PX_LEN)

# Fonts
font_big = pygame.font.SysFont("monospace", 22, bold=True)
font_med = pygame.font.SysFont("monospace", 16)
font_sml = pygame.font.SysFont("monospace", 13)

# Colours
BG = (15, 17, 26)
BOB_UP_C = (80, 255, 140)  # green — used for HUD text when upright
TORQUE_C = (255, 180, 50)
TEXT_C = (200, 210, 230)
WARN_C = (255, 80, 80)
GRID_C = (30, 35, 50)


def draw_pendulum(
    theta, theta_dot, torque, episode, step, reward_total, epsilon, training, fps_actual
):
    # Draw the pendulum into the off-screen surface, then blit to screen
    renderer.draw(theta, theta_dot, torque)
    screen.blit(renderer.surface, (0, 0))

    upright = abs(theta) < 0.2

    # --- grid lines (drawn on screen after blit so they appear on top,
    #     colour is dark enough that it doesn't obscure the pendulum) ---
    for r in range(50, 300, 50):
        pygame.draw.circle(screen, GRID_C, (CX, CY), r, 1)
    for angle_deg in range(0, 360, 30):
        rad = math.radians(angle_deg)
        pygame.draw.line(
            screen,
            GRID_C,
            (CX, CY),
            (CX + int(250 * math.cos(rad)), CY + int(250 * math.sin(rad))),
            1,
        )

    # --- upright zone arc ---
    pygame.draw.arc(
        screen,
        (40, 80, 40),
        (CX - PX_LEN, CY - PX_LEN, 2 * PX_LEN, 2 * PX_LEN),
        math.pi / 2 - 0.2,
        math.pi / 2 + 0.2,
        4,
    )

    # --- torque indicator arc ---
    if torque != 0:
        arc_r = 30
        start = -math.pi / 2 + (0 if torque < 0 else math.pi)
        end = start + math.pi * (torque / MAX_TORQUE)
        if start > end:
            start, end = end, start
        pygame.draw.arc(
            screen,
            TORQUE_C,
            (CX - arc_r, CY - arc_r, 2 * arc_r, 2 * arc_r),
            start,
            end,
            3,
        )

    # --- HUD ---------------------------------------------------------------
    y0 = 360
    lh = 22

    def txt(label, value, col=TEXT_C, y_offset=0):
        nonlocal y0
        surf = font_med.render(f"{label:<18}{value}", True, col)
        screen.blit(surf, (30, y0 + y_offset))
        y0 += lh

    mode_str = "TRAINING" if training else "WATCHING"
    mode_col = WARN_C if training else BOB_UP_C
    mode_surf = font_big.render(mode_str, True, mode_col)
    screen.blit(mode_surf, (W - mode_surf.get_width() - 20, H - mode_surf.get_height() - 20))

    y0 = 360
    txt("Episode:", f"{episode}")
    txt("Step:", f"{step} / {MAX_STEPS}")
    txt("Upright time:", f"{reward_total:.0f} steps")
    txt("Epsilon:", f"{epsilon:.3f}")
    txt("FPS:", f"{fps_actual:.0f}")

    hint1 = font_sml.render(
        "SPACE: toggle train/watch   R: reset   Q: quit", True, (100, 110, 130)
    )
    screen.blit(hint1, (30, H - 30))

    pygame.display.flip()


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    env = Pendulum()
    epsilon = EPSILON_START
    episode = 0
    training = True  # start in training mode
    fps_actual = 0.0

    theta, theta_dot = env.reset(
        theta=math.pi + np.random.uniform(-0.3, 0.3),
        theta_dot=np.random.uniform(-0.5, 0.5),
    )
    step = 0
    reward_total = 0.0
    last_torque = 0.0

    t_last = time.perf_counter()

    while True:
        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    theta, theta_dot = env.reset(
                        theta=math.pi + np.random.uniform(-0.3, 0.3),
                        theta_dot=np.random.uniform(-0.5, 0.5),
                    )
                    step = 0
                    reward_total = 0.0
                if event.key == pygame.K_SPACE:
                    training = not training

        # ── Agent step ──
        a_idx, s_idx = discretise(theta, theta_dot)
        act_idx = choose_action(a_idx, s_idx, epsilon if training else 0.0)
        torque = ACTIONS[act_idx]
        last_torque = torque

        theta_new, theta_dot_new = env.step(torque)
        reward = 1.0 if abs(theta_new) < 0.2 else 0.0
        reward_total += reward
        step += 1

        if training:
            a2, s2 = discretise(theta_new, theta_dot_new)
            update_q(a_idx, s_idx, act_idx, reward, a2, s2)

        theta, theta_dot = theta_new, theta_dot_new

        # ── Episode end ──
        if step >= MAX_STEPS:
            if training:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            episode += 1
            theta, theta_dot = env.reset(
                theta=math.pi + np.random.uniform(-0.3, 0.3),
                theta_dot=np.random.uniform(-0.5, 0.5),
            )
            step = 0
            reward_total = 0.0

        # ── Render ──
        draw_pendulum(
            theta,
            theta_dot,
            last_torque,
            episode,
            step,
            reward_total,
            epsilon,
            training,
            fps_actual,
        )

        # ── Timing ──
        if training:
            clock.tick(0)  # uncapped – train as fast as possible
        else:
            clock.tick(FPS_WATCH)  # smooth 60 fps for watching

        now = time.perf_counter()
        fps_actual = 1.0 / max(now - t_last, 1e-6)
        t_last = now


if __name__ == "__main__":
    main()
