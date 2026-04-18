"""Generic tabular Q-learning agent.

Knows nothing about the environment's physics or observation space.
State discretisation is injected as a callable so the agent works with
any discrete or discretised state representation.
"""

from collections.abc import Callable

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        q_shape: tuple,  # one int per state dimension, e.g. (32, 16)
        n_actions: int,
        discretize: Callable,  # obs -> state tuple, e.g. (angle_bin, speed_bin)
        alpha: float = 0.15,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self._discretize = discretize
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self.Q = np.zeros((*q_shape, n_actions))

    def act(self, obs, explore: bool = True) -> int:
        """Return an action index, epsilon-greedy when explore=True."""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[self._discretize(obs)]))

    def learn(self, obs, act_idx: int, reward: float, next_obs) -> None:
        """Single TD update."""
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        best_next = np.max(self.Q[next_state])
        self.Q[(*state, act_idx)] += self.alpha * (
            reward + self.gamma * best_next - self.Q[(*state, act_idx)]
        )

    def end_episode(self) -> None:
        """Decay epsilon at the end of a training episode."""
        self.epsilon = max(self._epsilon_end, self.epsilon * self._epsilon_decay)
