"""Generic Q-value heatmap + action frequency row renderer.

Displays a 2-D Q-table (states × actions) as a brightness heatmap with a thin
frequency strip directly below showing how often each action was taken this episode.

State-axis convention: frac=0 is the top of the heatmap, frac=1 is the bottom.
The caller is responsible for mapping environment state to that fraction and for
collapsing any extra Q-table dimensions down to (n_states, n_actions).
"""

import numpy as np
import pygame


class PolicyRenderer:
    _PAD = 40  # inner margin
    _BELOW = 70  # pixels reserved below heatmap for freq row + labels
    _FREQ_H = 12  # height of the frequency strip

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        action_range: tuple,  # (min_value, max_value) shown on X-axis
        action_name: str,  # X-axis label, e.g. "torque"
        state_ticks: list,  # [(frac_0_to_1, label_str), ...] for Y-axis
        action_ticks: list
        | None = None,  # [(frac, label), ...] for X-axis; auto if None
        q_title: str = "Q-value  (brighter = higher)",
        freq_title: str = "applied torque  (brighter=most frequent)",
        bg: tuple = (15, 17, 26),
        text_c: tuple = (200, 210, 230),
        hint_c: tuple = (90, 100, 120),
        current_c: tuple = (255, 180, 50),
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_range = action_range
        self.action_name = action_name
        self.state_ticks = state_ticks
        if action_ticks is None:
            a_min, a_max = action_range
            span = a_max - a_min
            action_ticks = [
                (
                    i / 4,
                    "0"
                    if (a_min + i * span / 4) == 0
                    else f"{a_min + i * span / 4:+.0f}",
                )
                for i in range(5)
            ]
        self.action_ticks = action_ticks
        self.q_title = q_title
        self.freq_title = freq_title
        self.bg = bg
        self.text_c = text_c
        self.hint_c = hint_c
        self.current_c = current_c
        self._font = pygame.font.SysFont("monospace", 13)

    def draw(
        self,
        surface: pygame.Surface,
        rect: tuple,  # (x, y, w, h)
        q2d: np.ndarray,  # (n_states, n_actions)
        state_frac: float,  # 0=top … 1=bottom, where to draw the marker
        action_counts: np.ndarray,  # (n_actions,) int counts this episode
        current_act: int,  # index of the action just taken
        state_label: str = "",  # optional text shown next to the marker line
    ) -> None:
        rx, ry, rw, rh = rect
        pad, below, freq_h = self._PAD, self._BELOW, self._FREQ_H

        pygame.draw.rect(surface, self.bg, rect)
        pygame.draw.line(surface, self.hint_c, (rx, ry), (rx, ry + rh), 1)

        cell_h = (rh - 2 * pad) // self.n_states
        map_h = self.n_states * cell_h
        map_w = rw - 2 * pad
        ox = rx + pad
        oy = max(ry + pad // 2, ry + rh - map_h - below)

        # ── Q-value heatmap ───────────────────────────────────────────────────
        q_min, q_max = q2d.min(), q2d.max()
        q_norm = (
            (q2d - q_min) / (q_max - q_min) if q_max > q_min else np.zeros_like(q2d)
        )
        intensity = (q_norm.T * 255).astype(np.uint8)  # (n_actions, n_states)
        rgb = np.stack([intensity, intensity, intensity], axis=2)[:, ::-1, :]
        scaled = pygame.transform.scale(
            pygame.surfarray.make_surface(rgb), (map_w, map_h)
        )
        surface.blit(scaled, (ox, oy))

        lbl = self._font.render(self.q_title, True, self.text_c)
        surface.blit(lbl, (ox + map_w // 2 - lbl.get_width() // 2, ry + 8))

        # Y-axis ticks
        for frac, label in self.state_ticks:
            yp = oy + int(frac * map_h)
            lbl = self._font.render(label, True, self.hint_c)
            surface.blit(lbl, (ox - lbl.get_width() - 4, yp - lbl.get_height() // 2))
            pygame.draw.line(surface, self.hint_c, (ox - 3, yp), (ox, yp), 1)

        # State marker
        ym = oy + int(state_frac * map_h)
        pygame.draw.line(surface, (255, 200, 0), (ox, ym), (ox + map_w, ym), 1)
        if state_label:
            lbl = self._font.render(state_label, True, (255, 200, 0))
            surface.blit(lbl, (ox + map_w + 4, ym - lbl.get_height() // 2))

        # Q heatmap X-axis ticks
        q_label_y = oy + map_h + 6
        for frac, label in self.action_ticks:
            xp = ox + int(frac * map_w)
            pygame.draw.line(
                surface, self.hint_c, (xp, oy + map_h), (xp, oy + map_h + 4), 1
            )
            lbl = self._font.render(label, True, self.hint_c)
            surface.blit(lbl, (xp - lbl.get_width() // 2, q_label_y))

        # ── Frequency row ─────────────────────────────────────────────────────
        freq_title_lbl = self._font.render(self.freq_title, True, self.text_c)
        title_y = q_label_y + self._font.get_height() + 4
        surface.blit(
            freq_title_lbl, (ox + map_w // 2 - freq_title_lbl.get_width() // 2, title_y)
        )

        freq_y = title_y + freq_title_lbl.get_height() + 2
        pygame.draw.rect(surface, (40, 42, 54), (ox, freq_y, map_w, freq_h))
        max_count = max(action_counts.max(), 1)
        for i, count in enumerate(action_counts):
            x0 = ox + i * map_w // self.n_actions
            x1 = ox + (i + 1) * map_w // self.n_actions
            iv = int(255 * count / max_count)
            color = self.current_c if i == current_act else (iv, iv, iv)
            pygame.draw.rect(surface, color, (x0, freq_y, x1 - x0, freq_h))

        # Freq row X-axis labels — centered on the same x positions as the Q-heatmap ticks
        label_y = freq_y + freq_h + 4
        a_min, a_max = self.action_range
        lbl = self._font.render(f"{a_min:.0f}", True, self.hint_c)
        surface.blit(lbl, (ox - lbl.get_width() // 2, label_y))
        lbl = self._font.render(f"+{a_max:.0f}", True, self.hint_c)
        surface.blit(lbl, (ox + map_w - lbl.get_width() // 2, label_y))
        lbl = self._font.render(self.action_name, True, self.hint_c)
        surface.blit(lbl, (ox + map_w // 2 - lbl.get_width() // 2, label_y))
