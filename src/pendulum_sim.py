"""Pendulum simulation — implements the Simulation protocol.

Contains everything specific to the inverted-pendulum problem:
physics renderer, reward shaping, state discretisation, panel drawing,
and CLI argument parsing.
"""

import argparse
import math

import numpy as np
import pygame

from pendulum import MAX_TORQUE, PendulumRenderer
from pendulum_env import PendulumEnv
from policy_renderer import PolicyRenderer
from q_agent import QLearningAgent
from simulation import Simulation

# ── Layout ────────────────────────────────────────────────────────────────────
_W_SIM, _W_POLICY, _H = 600, 400, 550
_CX, _CY = _W_SIM // 2, 260
_PX_LEN = 180

# ── Colours ───────────────────────────────────────────────────────────────────
_BG = (15, 17, 26)
_HINT_C = (90, 100, 120)
_BOB_UP_C = (80, 255, 140)
_TORQUE_C = (255, 180, 50)
_TEXT_C = (200, 210, 230)
_WARN_C = (255, 80, 80)
_GRID_C = (30, 35, 50)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_N_ANGLE = 32
_N_SPEED = 16
_MAX_SPEED = 20.0
_ACTIONS = np.linspace(-MAX_TORQUE, MAX_TORQUE, 20)

_angle_bins = np.linspace(-math.pi, math.pi, _N_ANGLE + 1)
_speed_bins = np.linspace(-_MAX_SPEED, _MAX_SPEED, _N_SPEED + 1)


def _make_discretize(use_speed: bool):
    def discretize(obs):
        theta, theta_dot = obs
        a = int(np.clip(np.digitize(theta, _angle_bins) - 1, 0, _N_ANGLE - 1))
        if use_speed:
            v = int(np.clip(np.digitize(theta_dot, _speed_bins) - 1, 0, _N_SPEED - 1))
            return (a, v)
        return (a,)

    return discretize


class PendulumSim(Simulation):
    def __init__(self, use_speed: bool = True, better_reward: bool = False):
        self._use_speed = use_speed

        # Required Simulation attributes
        self.env = PendulumEnv(better_reward=better_reward)
        self.actions = _ACTIONS
        self.max_steps = 500
        self.fps = 60
        self.window_size = (_W_SIM + _W_POLICY, _H)
        self.sim_rect = (0, 0, _W_SIM, _H)
        self.policy_rect = (_W_SIM, 0, _W_POLICY, _H)

        q_shape = (_N_ANGLE, _N_SPEED) if use_speed else (_N_ANGLE,)
        self.agent = QLearningAgent(
            q_shape=q_shape,
            n_actions=len(_ACTIONS),
            discretize=_make_discretize(use_speed),
        )

        self.policy_renderer = PolicyRenderer(
            n_states=_N_ANGLE,
            n_actions=len(_ACTIONS),
            action_range=(-MAX_TORQUE, MAX_TORQUE),
            action_name="torque",
            state_ticks=[
                (0.0, "+pi  "),
                (0.25, "+pi/2"),
                (0.5, "  0  "),
                (0.75, "-pi/2"),
                (1.0, "-pi  "),
            ],
        )

        self._pend_renderer = PendulumRenderer(
            width=_W_SIM, height=_H, cx=_CX, cy=_CY, scale=_PX_LEN, show_hud=True
        )
        self._fonts = {
            "big": pygame.font.SysFont("monospace", 22, bold=True),
            "med": pygame.font.SysFont("monospace", 16),
            "sml": pygame.font.SysFont("monospace", 13),
        }

    # ── Simulation protocol ───────────────────────────────────────────────────

    def render_panel(
        self, surface, last_act_idx, episode, step, reward_total, training, sps_actual
    ):
        last_torque = self.actions[last_act_idx]

        self._pend_renderer.draw(
            self.env.theta, self.env.theta_dot, last_torque, self.env.theta_ddot
        )
        surface.blit(self._pend_renderer.surface, (0, 0))

        for r in range(50, 300, 50):
            pygame.draw.circle(surface, _GRID_C, (_CX, _CY), r, 1)
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            pygame.draw.line(
                surface,
                _GRID_C,
                (_CX, _CY),
                (_CX + int(250 * math.cos(rad)), _CY + int(250 * math.sin(rad))),
                1,
            )

        pygame.draw.arc(
            surface,
            (40, 80, 40),
            (_CX - _PX_LEN, _CY - _PX_LEN, 2 * _PX_LEN, 2 * _PX_LEN),
            math.pi / 2 - 0.2,
            math.pi / 2 + 0.2,
            4,
        )

        if last_torque != 0:
            arc_r = 30
            start = -math.pi / 2 + (0 if last_torque < 0 else math.pi)
            end = start + math.pi * (last_torque / MAX_TORQUE)
            if start > end:
                start, end = end, start
            pygame.draw.arc(
                surface,
                _TORQUE_C,
                (_CX - arc_r, _CY - arc_r, 2 * arc_r, 2 * arc_r),
                start,
                end,
                3,
            )

        mode_str = "TRAINING" if training else "WATCHING"
        mode_col = _WARN_C if training else _BOB_UP_C
        mode_surf = self._fonts["big"].render(mode_str, True, mode_col)
        surface.blit(
            mode_surf,
            (_W_SIM - mode_surf.get_width() - 20, _H - mode_surf.get_height() - 20),
        )

        lh, y0 = 22, 330

        def txt(label, value, col=_TEXT_C):
            nonlocal y0
            surface.blit(
                self._fonts["med"].render(f"{label:<14}{value}", True, col), (30, y0)
            )
            y0 += lh

        txt("Episode:", f"{episode}")
        txt("Step:", f"{step}/{self.max_steps}")
        txt("State:", "angle+speed" if self._use_speed else "angle only", col=_HINT_C)
        txt("Upright:", f"{reward_total:.0f} steps")
        txt("Epsilon:", f"{self.agent.epsilon:.3f}")
        txt("SPS:", f"{sps_actual:.0f}")

        surface.blit(
            self._fonts["sml"].render(
                "SPACE: train/watch   R: reset   Q: quit", True, _HINT_C
            ),
            (30, _H - 30),
        )

    def state_frac(self, obs) -> float:
        theta, _ = obs
        return (math.pi - theta) / (2 * math.pi)

    def state_label(self, obs) -> str:
        theta, _ = obs
        return f"{theta:+.2f}"

    def q2d(self) -> np.ndarray:
        return self.agent.Q.max(axis=1) if self._use_speed else self.agent.Q


# ── Factory ───────────────────────────────────────────────────────────────────


def make_pendulum_sim() -> PendulumSim:
    """Parse pendulum-specific CLI args and return a ready-to-run PendulumSim."""
    parser = argparse.ArgumentParser(description="Q-Learning Pendulum Balancer")
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Angle-only Q-table (ignore angular velocity).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--simple-reward", action="store_true")
    group.add_argument(
        "--better-reward",
        action="store_true",
        help="Gentleness bonus to reduce jitter.",
    )
    args = parser.parse_args()
    return PendulumSim(use_speed=not args.no_speed, better_reward=args.better_reward)
