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
MAX_TORQUE = 40.0  # N·m
MAX_ROD_FORCE = 200.0  # N (rod breaking point)


class Pendulum:
    """Single rigid-rod pendulum with Euler integration.

    State:
        theta     – angle in radians (0 = upright, ±π = hanging)
        theta_dot – angular velocity (rad/s)
    """

    def __init__(self, max_rod_force: float = MAX_ROD_FORCE):
        """Initialize pendulum at rest hanging position."""
        self.theta = math.pi  # start hanging
        self.theta_dot = 0.0
        self.theta_ddot = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.terminated = False
        self.max_rod_force = max_rod_force

    def reset(self, theta=math.pi, theta_dot=0.0):
        """Set state directly; defaults to hanging at rest.

        Returns:
            (theta, theta_dot) – initial angle (rad) and angular velocity (rad/s)
        """
        self.theta = (float(theta) + math.pi) % (2 * math.pi) - math.pi
        self.theta_dot = float(theta_dot)
        self.terminated = False
        return self.theta, self.theta_dot

    def step(self, torque):
        """Apply torque (N·m) for one timestep DT.

        Returns:
            (ax, ay, terminated)
            ax         – bob acceleration in x (m/s²) — what an accelerometer reads
            ay         – bob acceleration in y (m/s²) — what an accelerometer reads
            terminated – True if rod force exceeds MAX_ROD_FORCE (rod breaks)

        Note: theta, theta_dot, theta_ddot are internal state; access via
        self.theta / self.theta_dot if needed for state discretization.
        """
        th, td = self.theta, self.theta_dot
        self.theta_ddot = (
            (G / L) * math.sin(th)  # gravity: pulls toward bottom
            - (B / (M * L**2)) * td  # damping: opposes motion (friction at pivot)
            + torque / (M * L**2)  # applied torque
        )
        self.theta_dot = td + self.theta_ddot * DT
        self.theta = (th + self.theta_dot * DT + math.pi) % (2 * math.pi) - math.pi

        # Rod tension: gravity component + centripetal force
        rod_force = M * (G * math.cos(self.theta) + L * self.theta_dot**2)
        self.terminated = rod_force > self.max_rod_force

        # Proper acceleration (internal, available for future sensor simulation)
        self.ax = L * (
            -math.sin(self.theta) * self.theta_dot**2
            + math.cos(self.theta) * self.theta_ddot
        )
        self.ay = (
            L
            * (
                math.cos(self.theta) * self.theta_dot**2
                + math.sin(self.theta) * self.theta_ddot
            )
            - G
        )
        return self.theta, self.theta_dot, self.theta_ddot, self.terminated


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

    def draw(self, theta, theta_dot, torque, theta_ddot=0.0):
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
            lh = 20
            x, y = 12, 12

            def row(label, value, col):
                nonlocal y
                surf.blit(self.font.render(f"{label:<12}{value}", True, col), (x, y))
                y += lh

            rod_force = M * (G * math.cos(theta) + L * theta_dot**2)
            rod_col = self.BOB_UP_C if rod_force < MAX_ROD_FORCE else self.THETA_C
            angle_col = self.BOB_UP_C if upright else self.THETA_C
            row("theta:", f"{theta:+.4f} rad", angle_col)
            row("theta_dot:", f"{theta_dot:+.2f} rad/s", self.TEXT_C)
            row("theta_ddot:", f"{theta_ddot:+.2f} rad/s²", self.TEXT_C)
            row("rod_force:", f"{rod_force:+.1f} N", rod_col)
            row("torque:", f"{torque:+.1f} N·m", self.TORQUE_C)

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
    pend.reset(theta=math.pi, theta_dot=0.0)
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
                    pend.reset(theta=math.pi, theta_dot=0.0)
                    terminated = False

        if not terminated:
            torque = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                torque = +MAX_TORQUE
            if keys[pygame.K_LEFT]:
                torque = -MAX_TORQUE
            _theta, _theta_dot, _theta_ddot, terminated = pend.step(torque)

        renderer.draw(pend.theta, pend.theta_dot, torque, pend.theta_ddot)
        screen.blit(renderer.surface, (0, 0))

        if terminated:
            msg = font.render(
                f"BROKEN: rod force exceeded {MAX_ROD_FORCE:.0f} N — press R",
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
