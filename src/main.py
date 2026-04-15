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
N_ANGLE = 32  # angle bins   (–π … π)
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


def discretise(theta):
    """Return a state tuple (angle_bin,) used to index the Q-table."""
    a = int(np.clip(np.digitize(theta, angle_bins) - 1, 0, N_ANGLE - 1))
    return (a,)


# ── Q-table ────────────────────────────────────────────────────────────────────
Q = np.zeros((N_ANGLE, N_ACTS))


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTS)
    return int(np.argmax(Q[state]))


def update_q(state, act, reward, next_state):
    best_next = np.max(Q[next_state])
    Q[state + (act,)] += ALPHA * (reward + GAMMA * best_next - Q[state + (act,)])


# ── Policy heatmap ────────────────────────────────────────────────────────────
def draw_heatmap(theta):
    """Render Q-values as action vs angle in the right panel.

    X axis: angle bins  (–π … π, left to right)
    Y axis: action/torque (max negative at bottom, max positive at top)
    Colour: Q-value intensity — bright = high value, dark = low
    Greedy action at each angle is highlighted in white.
    """
    pygame.draw.rect(screen, BG, (W_PEND, 0, W_HEAT, H))

    q2d = Q  # (N_ANGLE, N_ACTS)

    # Normalise Q-values to [0, 1] for brightness
    q_min, q_max = q2d.min(), q2d.max()
    if q_max > q_min:
        q_norm = (q2d - q_min) / (q_max - q_min)
    else:
        q_norm = np.zeros_like(q2d)

    # Transpose to (N_ACTS, N_ANGLE): X = actions, Y = angle
    q2d_t = q2d.T

    # Brightness = Q-value; greedy action is naturally the brightest cell
    intensity = (q_norm.T * 255).astype(np.uint8)
    r = intensity
    g = intensity
    b = intensity

    # rgb shape (N_ACTS, N_ANGLE, 3); flip Y so +pi is at top
    rgb = np.stack([r, g, b], axis=2)[:, ::-1, :]

    # Scale up to fill the heatmap panel
    pad = 40
    cell = min((W_HEAT - 2 * pad) // N_ACTS, (H - 2 * pad) // N_ANGLE)
    map_w, map_h = N_ACTS * cell, N_ANGLE * cell
    ox = W_PEND + (W_HEAT - map_w) // 2
    oy = (H - map_h) // 2

    small = pygame.surfarray.make_surface(rgb)
    scaled = pygame.transform.scale(small, (map_w, map_h))
    screen.blit(scaled, (ox, oy))

    # Divider line between panels
    pygame.draw.line(screen, HINT_C, (W_PEND, 0), (W_PEND, H), 1)

    # X axis labels (actions / torque)
    lbl = font_sml.render(f"-{MAX_TORQUE:.0f}", True, HINT_C)
    screen.blit(lbl, (ox, oy + map_h + 4))
    lbl = font_sml.render(f"+{MAX_TORQUE:.0f}", True, HINT_C)
    screen.blit(lbl, (ox + map_w - lbl.get_width(), oy + map_h + 4))
    lbl = font_sml.render("torque", True, HINT_C)
    screen.blit(lbl, (ox + map_w // 2 - lbl.get_width() // 2, oy + map_h + 4))

    # Y axis labels (angle in radians): +pi at top, -pi at bottom
    angle_ticks = [
        (math.pi, "+pi  "),
        (math.pi / 2, "+pi/2"),
        (0.0, "  0  "),
        (-math.pi / 2, "-pi/2"),
        (-math.pi, "-pi  "),
    ]
    for angle_val, label in angle_ticks:
        # Map angle to pixel y: top=+pi, bottom=-pi
        frac = (math.pi - angle_val) / (2 * math.pi)
        y_pos = oy + int(frac * map_h)
        lbl = font_sml.render(label, True, HINT_C)
        screen.blit(lbl, (ox - lbl.get_width() - 4, y_pos - lbl.get_height() // 2))
        pygame.draw.line(screen, HINT_C, (ox - 3, y_pos), (ox, y_pos), 1)

    # Current theta marker — horizontal line across the heatmap
    frac = (math.pi - theta) / (2 * math.pi)
    y_theta = oy + int(frac * map_h)
    pygame.draw.line(screen, (255, 200, 0), (ox, y_theta), (ox + map_w, y_theta), 1)
    lbl = font_sml.render(f"{theta:+.2f}", True, (255, 200, 0))
    screen.blit(lbl, (ox + map_w + 4, y_theta - lbl.get_height() // 2))

    # Title
    title = font_sml.render("Q-value  (brighter = higher)", True, TEXT_C)
    screen.blit(title, (ox + map_w // 2 - title.get_width() // 2, 8))


# ── Pygame visualisation ───────────────────────────────────────────────────────
W_PEND, W_HEAT, H = 600, 400, 550
W = W_PEND + W_HEAT  # total window width
CX, CY = W_PEND // 2, 260  # pivot centre (within the pendulum panel)
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
renderer = PendulumRenderer(
    width=W, height=H, cx=CX, cy=CY, scale=PX_LEN, show_hud=True
)

# Fonts
font_big = pygame.font.SysFont("monospace", 22, bold=True)
font_med = pygame.font.SysFont("monospace", 16)
font_sml = pygame.font.SysFont("monospace", 13)

# Colours
BG = (15, 17, 26)
HINT_C = (90, 100, 120)
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
    screen.blit(
        mode_surf, (W - mode_surf.get_width() - 20, H - mode_surf.get_height() - 20)
    )

    y0 = 390
    txt("Episode:", f"{episode}")
    txt("Step:", f"{step} / {MAX_STEPS}")
    txt("Upright time:", f"{reward_total:.0f} steps")
    txt("Epsilon:", f"{epsilon:.3f}")
    txt("FPS:", f"{fps_actual:.0f}")

    hint1 = font_sml.render(
        "SPACE: train/watch   R: reset   Q: quit", True, (100, 110, 130)
    )
    screen.blit(hint1, (30, H - 30))


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    env = Pendulum()
    epsilon = EPSILON_START
    episode = 0
    training = True  # start in training mode
    fps_actual = 0.0

    theta = env.reset(
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
                    theta = env.reset(
                        theta=math.pi + np.random.uniform(-0.3, 0.3),
                        theta_dot=np.random.uniform(-0.5, 0.5),
                    )
                    step = 0
                    reward_total = 0.0
                if event.key == pygame.K_SPACE:
                    training = not training

        # ── Agent step ──
        state = discretise(theta)
        act_idx = choose_action(state, epsilon if training else 0.0)
        torque = ACTIONS[act_idx]
        last_torque = torque

        theta_new, terminated = env.step(torque)
        if terminated:
            reward = -1.0
        else:
            reward = 1.0 if abs(theta_new) < 0.2 else 0.0
        reward_total += reward
        step += 1

        if training:
            next_state = discretise(theta_new)
            update_q(state, act_idx, reward, next_state)

        theta = theta_new

        # ── Episode end ──
        if terminated or step >= MAX_STEPS:
            if training:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            episode += 1
            theta = env.reset(
                theta=math.pi + np.random.uniform(-0.3, 0.3),
                theta_dot=np.random.uniform(-0.5, 0.5),
            )
            step = 0
            reward_total = 0.0

        # ── Render ──
        draw_pendulum(
            theta,
            env.theta_dot,
            last_torque,
            episode,
            step,
            reward_total,
            epsilon,
            training,
            fps_actual,
        )
        draw_heatmap(theta)
        pygame.display.flip()

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
