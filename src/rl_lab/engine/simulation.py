"""Abstract base class for simulations compatible with the generic game loop."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pygame


class Simulation(ABC):
    # Subclasses must assign these in __init__
    env: Any  # reset() -> obs,  step(action) -> (obs, reward, done)
    agent: Any  # act(), learn(), end_episode(), epsilon, n_actions
    actions: np.ndarray  # index -> action value passed to env.step()
    policy_renderer: Any  # configured PolicyRenderer
    max_steps: int
    fps: int  # frame rate in watch mode
    window_size: tuple  # (W, H) for pygame.display.set_mode()
    sim_rect: tuple  # (x, y, w, h) simulation panel
    policy_rect: tuple  # (x, y, w, h) policy panel

    @abstractmethod
    def render_panel(
        self,
        surface: pygame.Surface,
        last_act_idx: int,
        episode: int,
        step: int,
        reward_total: float,
        training: bool,
        sps_actual: float,
    ) -> None:
        """Draw the simulation-specific left panel onto surface."""

    @abstractmethod
    def state_frac(self, obs) -> float:
        """Map obs to [0, 1] for the policy renderer's state-marker line (0=top, 1=bottom)."""

    @abstractmethod
    def state_label(self, obs) -> str:
        """Short string shown next to the state-marker line on the policy panel."""

    @abstractmethod
    def q2d(self) -> np.ndarray:
        """Return a 2-D (n_states × n_actions) view of the Q-table for the policy renderer."""
