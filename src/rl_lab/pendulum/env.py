"""Gym-style wrapper around the Pendulum physics.

Owns the reward function and episode reset logic so the main loop
stays agnostic to physics details and reward shaping choices.
"""

import math

import numpy as np

from rl_lab.pendulum.physics import MAX_TORQUE, Pendulum


class PendulumEnv:
    def __init__(self, better_reward: bool = False):
        self._phys = Pendulum()
        self.better_reward = better_reward
        self.theta: float = math.pi
        self.theta_dot: float = 0.0
        self.theta_ddot: float = 0.0

    def reset(self) -> tuple:
        """Reset to a random near-hanging position. Returns (theta, theta_dot)."""
        theta = math.pi + np.random.uniform(-0.3, 0.3)
        theta_dot = np.random.uniform(-0.5, 0.5)
        self.theta, self.theta_dot = self._phys.reset(theta=theta, theta_dot=theta_dot)
        self.theta_ddot = 0.0
        return (self.theta, self.theta_dot)

    def step(self, torque: float) -> tuple:
        """Advance one timestep. Returns ((theta, theta_dot), reward, done)."""
        self.theta, self.theta_dot, self.theta_ddot, terminated = self._phys.step(
            torque
        )
        reward = self._reward(torque, terminated)
        return (self.theta, self.theta_dot), reward, terminated

    def _reward(self, torque: float, terminated: bool) -> float:
        if terminated:
            return -1.0
        upright = abs(self.theta) < 0.2
        if self.better_reward:
            if upright:
                # more reward for being close to upright (0 rads), and for being gentle (low speed and low torque)
                gentleness = max(
                    0.0, 1.0 - self.theta_dot**2 - (torque / MAX_TORQUE) ** 2
                )
                closer_to_upright = 1.0 - abs(self.theta) / math.pi
                return 1.0 + gentleness + 2 * closer_to_upright
            return 0.0
        return 1.0 if upright else 0.0
