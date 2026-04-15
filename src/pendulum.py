"""
Inverted pendulum — pure physics module.

Physics:
  θ̈ = (g/L)·sin(θ) - (b/mL²)·θ̇ + (1/mL²)·τ
        ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^
        gravity         damping         applied torque
        pulls to bottom opposes motion  agent control
                        (pivot friction)

  θ = 0        → upright (top)
  θ = ±π       → hanging (bottom)

Run directly to test the physics interactively:
  python pendulum.py

  LEFT / RIGHT arrow keys  - apply torque
  R                        — reset to hanging position
  Q / ESC                  — quit
"""

import math
import sys

# ── Physics constants ──────────────────────────────────────────────────────────
G = 9.81  # gravity   (m/s²)
L = 1.0  # length    (m)
M = 1.0  # mass      (kg)
B = 0.05  # damping   (N·m·s)
DT = 0.02  # timestep  (s)
MAX_TORQUE = 20.0  # N·m


class Pendulum:
    """Single rigid-rod pendulum with Euler integration.

    State:
        theta     – angle in radians (0 = upright, ±π = hanging)
        theta_dot – angular velocity (rad/s)
    """

    def __init__(self, max_speed=20.0):
        """
        max_speed – angular velocity limit (rad/s). If exceeded, the pendulum
                    is considered broken and step() sets terminated=True.
                    Prevents the agent from exploiting fast spinning through
                    the upright zone.
        """
        self.max_speed = max_speed
        self.theta = math.pi  # start hanging
        self.theta_dot = 0.0
        self.terminated = False

    def reset(self, theta=math.pi, theta_dot=0.0):
        """Set state directly; defaults to hanging at rest.

        Returns:
            theta – initial angle (rad), wrapped to (−π, π]
        """
        self.theta = (float(theta) + math.pi) % (2 * math.pi) - math.pi
        self.theta_dot = float(theta_dot)
        self.terminated = False
        return self.theta

    def step(self, torque):
        """Apply torque (N·m) for one timestep DT.

        Returns:
            theta      – new angle (rad), wrapped to (−π, π]
            terminated – True if the pendulum broke (|θ̇| > max_speed)
        """
        th, td = self.theta, self.theta_dot
        th_ddot = (
            (G / L) * math.sin(th)  # gravity: pulls toward bottom
            - (B / (M * L**2)) * td  # damping: opposes motion (friction at pivot)
            + torque / (M * L**2)  # applied torque
        )
        self.theta_dot = td + th_ddot * DT
        self.theta = (th + self.theta_dot * DT + math.pi) % (2 * math.pi) - math.pi
        self.terminated = abs(self.theta_dot) > self.max_speed
        return self.theta, self.terminated


class PendulumRenderer:
    """Renders a pendulum into an off-screen pygame Surface.

    Usage:
        # default: pivot centred in the surface
        renderer = PendulumRenderer(width=480, height=480)

        # custom pivot: leave room below for a HUD
        renderer = PendulumRenderer(width=600, height=550, cx=300, cy=200)

        renderer.draw(theta, theta_dot, torque)
        screen.blit(renderer.surface, (0, 0))   # display it
        frame = renderer.get_frame()            # numpy (H, W, 3) for CV
    """

    BG = (15, 17, 26)
    ROD_C = (160, 180, 220)
    PIVOT_C = (200, 200, 220)
    BOB_C = (80, 180, 255)
    BOB_UP_C = (80, 255, 140)  # green — upright bob and angle label
    TEXT_C = (200, 210, 230)
    HINT_C = (90, 100, 120)
    THETA_C = (255, 80, 80)  # red   — angle label when not upright
    TORQUE_C = (255, 180, 50)  # orange — torque label

    def __init__(
        self, width=480, height=480, cx=None, cy=None, scale=160, show_hud=False
    ):
        """
        width, height – size of the off-screen surface in pixels.

        cx, cy    – pixel position of the pivot (the fixed end of the rod)
                    within the surface. Defaults to the surface centre.
                    Adjust when you need room for a HUD below the pendulum,
                    e.g. cx=300, cy=200 on a 600x550 surface leaves space
                    at the bottom for Q-learning stats.

        scale     – pixels per metre; controls how long the rod appears.

        show_hud  – if True, overlays angle, angular velocity, torque, and
                    physics constants on the surface. Useful for standalone
                    testing; disable when the caller draws its own HUD.
        """
        import pygame

        self.W = width
        self.H = height
        self.CX = cx if cx is not None else width // 2  # pivot x
        self.CY = cy if cy is not None else height // 2  # pivot y
        self.PX_LEN = scale  # pixels per metre
        self.show_hud = show_hud
        self.PIVOT_R = 7
        self.BOB_R = 15
        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("monospace", 15)

    def draw(self, theta, theta_dot, torque):
        import pygame

        surf = self.surface
        surf.fill(self.BG)

        bx = self.CX + int(self.PX_LEN * math.sin(theta))
        by = self.CY - int(self.PX_LEN * math.cos(theta))

        upright = abs(theta) < 0.2
        pygame.draw.line(surf, self.ROD_C, (self.CX, self.CY), (bx, by), 4)
        pygame.draw.circle(surf, self.PIVOT_C, (self.CX, self.CY), self.PIVOT_R)
        pygame.draw.circle(
            surf, self.BOB_UP_C if upright else self.BOB_C, (bx, by), self.BOB_R
        )
        pygame.draw.circle(surf, (255, 255, 255), (bx, by), self.BOB_R, 2)

        if self.show_hud:
            # ── HUD top-left: dynamic state ──
            angle_col = self.BOB_UP_C if upright else self.THETA_C
            hud = [
                (f"theta:     {theta:+.4f} rad", angle_col),
                (f"theta_dot: {theta_dot:+7.2f} rad/s", self.TEXT_C),
                (f"torque:    {torque:+7.1f} N·m", self.TORQUE_C),
            ]
            for i, (line, col) in enumerate(hud):
                surf.blit(self.font.render(line, True, col), (12, 12 + i * 20))

            # ── HUD top-right: fixed physics constants ──
            consts = [
                f"L = {L:.2f} m",
                f"M = {M:.2f} kg",
            ]
            for i, line in enumerate(consts):
                s = self.font.render(line, True, self.HINT_C)
                surf.blit(s, (self.W - s.get_width() - 12, 12 + i * 20))

    def get_frame(self):
        """Return the current surface as a (H, W, 3) numpy array."""
        import numpy as np
        import pygame

        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)


# ── Interactive physics demo ───────────────────────────────────────────────────
def main():
    import pygame

    pygame.init()
    W, H = 480, 480
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pendulum — physics test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 15)

    HINT_C = (90, 100, 120)

    renderer = PendulumRenderer(width=W, height=H, show_hud=True)
    pend = Pendulum()
    theta = pend.reset(theta=math.pi, theta_dot=0.0)
    terminated = False
    torque = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    theta = pend.reset(theta=math.pi, theta_dot=0.0)
                    terminated = False

        if not terminated:
            torque = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                torque = +MAX_TORQUE
            if keys[pygame.K_LEFT]:
                torque = -MAX_TORQUE
            theta, terminated = pend.step(torque)

        renderer.draw(theta, pend.speed, torque)
        screen.blit(renderer.surface, (0, 0))

        if terminated:
            msg = font.render(
                f"BROKEN: speed exceeded {pend.max_speed:.0f} rad/s — press R",
                True,
                (255, 80, 80),
            )
            screen.blit(msg, ((W - msg.get_width()) // 2, H // 2))

        hint = font.render("LEFT/RIGHT: torque   R: reset   Q: quit", True, HINT_C)
        screen.blit(hint, (12, H - 26))

        pygame.display.flip()
        clock.tick(int(1 / DT))


if __name__ == "__main__":
    main()
