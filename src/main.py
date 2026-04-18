"""Generic Q-learning game loop.

Has no knowledge of any specific simulation — depends only on the
Simulation ABC. Entry point wires up a concrete simulation and calls run().

Controls (handled here, simulation-agnostic):
  SPACE  – toggle training / watching
  R      – reset episode
  Q/ESC  – quit
"""

import sys
import time

import numpy as np
import pygame

from simulation import Simulation


def run(sim: Simulation) -> None:
    screen = pygame.display.set_mode(sim.window_size)
    pygame.display.set_caption("Q-Learning")
    clock = pygame.time.Clock()

    training = True
    episode = 0
    fps_actual = 0.0

    obs = sim.env.reset()
    step = 0
    reward_total = 0.0
    last_act_idx = 0
    action_counts = np.zeros(sim.agent.n_actions, dtype=int)
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
                    obs = sim.env.reset()
                    step = 0
                    reward_total = 0.0
                    action_counts[:] = 0
                if event.key == pygame.K_SPACE:
                    training = not training

        # ── Step ──────────────────────────────────────────────────────────────
        act_idx = sim.agent.act(obs, explore=training)
        action = sim.actions[act_idx]
        last_act_idx = act_idx
        action_counts[act_idx] += 1

        next_obs, reward, done = sim.env.step(action)
        reward_total += reward
        step += 1

        if training:
            sim.agent.learn(obs, act_idx, reward, next_obs)
        obs = next_obs

        # ── Episode end ───────────────────────────────────────────────────────
        if done or step >= sim.max_steps:
            if training:
                sim.agent.end_episode()
            episode += 1
            obs = sim.env.reset()
            step = 0
            reward_total = 0.0
            action_counts[:] = 0

        # ── Render ────────────────────────────────────────────────────────────
        sim.render_panel(
            screen, last_act_idx, episode, step, reward_total, training, fps_actual
        )
        sim.policy_renderer.draw(
            screen,
            sim.policy_rect,
            sim.q2d(),
            sim.state_frac(obs),
            action_counts,
            last_act_idx,
            state_label=sim.state_label(obs),
        )
        pygame.display.flip()

        # ── Timing ────────────────────────────────────────────────────────────
        clock.tick(0 if training else sim.fps)
        now = time.perf_counter()
        fps_actual = 1.0 / max(now - t_last, 1e-6)
        t_last = now


if __name__ == "__main__":
    pygame.init()
    from pendulum_sim import make_pendulum_sim

    run(make_pendulum_sim())
