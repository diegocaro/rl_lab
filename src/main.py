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

from engine.simulation import Simulation


def run(sim: Simulation) -> None:
    screen = pygame.display.set_mode(sim.window_size)
    pygame.display.set_caption("Q-Learning")
    clock = pygame.time.Clock()

    _TRAIN_RENDER_INTERVAL = 1.0 / 30  # cap render at 30 fps during training

    training = True
    episode = 0
    steps_this_second = 0
    sps_actual = 0.0  # steps per second
    obs = sim.env.reset()
    step = 0
    reward_total = 0.0
    last_act_idx = 0
    action_counts = np.zeros(sim.agent.n_actions, dtype=int)
    t_last = time.perf_counter()
    t_render = t_last
    t_sps = t_last

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
        steps_this_second += 1

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

        # ── Render (time-gated during training) ───────────────────────────────
        now = time.perf_counter()
        if now - t_sps >= 1.0:
            sps_actual = steps_this_second / (now - t_sps)
            steps_this_second = 0
            t_sps = now

        if not training or (now - t_render) >= _TRAIN_RENDER_INTERVAL:
            sim.render_panel(
                screen, last_act_idx, episode, step, reward_total, training, sps_actual
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
            t_render = now

        # ── Timing ────────────────────────────────────────────────────────────
        if not training:
            clock.tick(sim.fps)
        t_last = now


if __name__ == "__main__":
    pygame.init()
    from pendulum.sim import make_pendulum_sim

    run(make_pendulum_sim())
