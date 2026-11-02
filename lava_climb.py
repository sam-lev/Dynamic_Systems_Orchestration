#!/usr/bin/env python3
# lava_climb.py -- Lava Climb: a real-time platformer on the live lava lamp
#
# Runs as a second process next to audible_lava.py.  The synth process
# publishes tracked blob outlines (and optionally the ROI video) over UDP
# via lava_bridge.py; this game turns every blob into a moving platform.
# Get the little blue climber from the base of the lamp to the top.
# Blobs rise, fall, split and merge in real time, so no two runs are the
# same and the music you hear *is* the level you are climbing.
#
# Usage:
#   python lava_climb.py            # listen on UDP 9100 for the live lamp
#   python lava_climb.py --sim      # synthetic blobs, no camera needed
#   python lava_climb.py --no-video # ignore the JPEG background
#   python lava_climb.py --port N
#
# Keys:
#   A/D or <-/->   move      SPACE / W / UP   jump (hold for higher)
#   R              restart   ESC / Q          quit
#
# Physics notes:
#   - Blobs are one-way platforms: you land on them from above and can
#     jump up through them.  A blob that carries you upward is the fast
#     way to the top; one that sinks takes you with it.
#   - Between UDP packets (15 Hz) polygons are extrapolated with the
#     tracker's Kalman velocity so platforms move smoothly at 60 fps.
#   - The lamp glass is the only wall.  Fall into the liquid below the
#     base and you respawn.

import base64
import io
import json
import math
import random
import socket
import sys
import time

import pygame

# ── Config ────────────────────────────────────────────────────────────────
PORT          = 9100
WIN_H         = 820                 # window height; width follows ROI aspect
FPS           = 60
GRAVITY       = 1900.0              # px/s^2 (screen space)
JUMP_V        = -620.0
JUMP_CUT      = 0.45                # release early -> shorter jump
MOVE_SPEED    = 260.0
AIR_CONTROL   = 0.85
LAND_SNAP     = 14                  # max px to pop up out of a blob on landing
COYOTE_TIME   = 0.10
PACKET_STALE  = 1.5                 # seconds without packets -> "no signal"
BASE_H        = 26                  # solid base platform at the bottom
GOAL_FRAC     = 0.06                # top 6 % of the lamp is the goal zone
PLAYER_W, PLAYER_H = 18, 26

COL_BG        = (14, 17, 22)
COL_GLASS     = (60, 70, 85)
COL_BASE      = (70, 56, 40)
COL_GOAL      = (255, 196, 60)
COL_TEXT      = (230, 230, 230)
COL_DIM       = (130, 130, 140)
COL_BLOB = {
    "rising":  (255, 120, 60),
    "falling": (120, 80, 255),
    "turning": (255, 60, 170),
}


# ── Blob feed ─────────────────────────────────────────────────────────────
class BlobFeed:
    """Receives LavaBridge datagrams; keeps last packet + arrival time."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.packet   = None
        self.recv_t   = 0.0
        self.video    = None        # pygame.Surface of latest JPEG
        self.count    = 0

    def poll(self):
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                latest = data
            except BlockingIOError:
                break
        if latest is None:
            return
        try:
            pkt = json.loads(latest.decode("utf-8"))
        except ValueError:
            return
        self.packet = pkt
        self.recv_t = time.time()
        self.count += 1
        jpg = pkt.get("jpg")
        if jpg:
            try:
                img = pygame.image.load(io.BytesIO(base64.b64decode(jpg)))
                self.video = img.convert()
            except pygame.error:
                pass

    @property
    def alive(self):
        return self.packet is not None and \
               time.time() - self.recv_t < PACKET_STALE


class SimFeed:
    """Synthetic lamp for playing without the camera.  Ellipsoid blobs
    cycle rise -> turn -> fall -> turn with a bit of wobble, plus an
    occasional split/merge so the geometry is not too friendly."""

    def __init__(self, w=320, h=640, n=6, seed=None):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.t = 0.0
        self.blobs = [self._new(i, self.rng.uniform(0.1, 0.9)) for i in range(n)]
        self.packet = None
        self.recv_t = 0.0
        self.video  = None
        self.count  = 0
        self.alive  = True

    def _new(self, bid, yfrac=None):
        return {
            "id": bid,
            "x": self.rng.uniform(40, self.w - 40),
            "y": (yfrac if yfrac is not None else 0.92) * self.h,
            "rx": self.rng.uniform(22, 46),
            "ry": self.rng.uniform(16, 34),
            "vy": -self.rng.uniform(14, 26),
            "phase": self.rng.uniform(0, 6.28),
            "wob": self.rng.uniform(0.4, 1.1),
        }

    def step(self, dt):
        self.t += dt
        blobs = []
        for b in self.blobs:
            # Buoyancy: rise to top, linger, sink, linger.
            if b["y"] < 0.08 * self.h and b["vy"] < 0:
                b["vy"] = +self.rng.uniform(10, 22)
                b["rx"], b["ry"] = b["ry"] * 1.3, b["rx"] * 0.8
            elif b["y"] > 0.92 * self.h and b["vy"] > 0:
                b["vy"] = -self.rng.uniform(14, 26)
                b["rx"], b["ry"] = b["ry"] * 1.2, b["rx"] * 0.85
            b["y"] += b["vy"] * dt
            vx = math.sin(self.t * b["wob"] + b["phase"]) * 12
            b["x"] = min(max(b["x"] + vx * dt, b["rx"] + 4), self.w - b["rx"] - 4)
            rx = b["rx"] * (1 + 0.08 * math.sin(self.t * 1.3 + b["phase"]))
            ry = b["ry"] * (1 + 0.08 * math.cos(self.t * 1.1 + b["phase"]))
            poly = [[b["x"] + rx * math.cos(a), b["y"] + ry * math.sin(a)]
                    for a in [i * 6.2832 / 18 for i in range(18)]]
            fps = 15.0
            blobs.append({
                "id": b["id"], "cx": b["x"], "cy": b["y"],
                "vx": vx / fps, "vy": b["vy"] / fps,
                "dir": "rising" if b["vy"] < -3 else "falling" if b["vy"] > 3 else "turning",
                "poly": poly,
            })
        self.packet = {"t": time.time(), "w": self.w, "h": self.h,
                       "fps": 15.0, "blobs": blobs}
        self.recv_t = time.time()
        self.count += 1

    def poll(self):
        pass


# ── Player ────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, x, y):
        self.reset(x, y)

    def reset(self, x, y):
        self.x, self.y = float(x), float(y)       # feet centre
        self.vx = self.vy = 0.0
        self.grounded   = False
        self.carrier_vy = 0.0
        self.carrier_vx = 0.0
        self.coyote     = 0.0
        self.facing     = 1
        self.jumping    = False

    @property
    def rect(self):
        return pygame.Rect(int(self.x - PLAYER_W / 2),
                           int(self.y - PLAYER_H), PLAYER_W, PLAYER_H)


# ── Game ──────────────────────────────────────────────────────────────────
class Game:
    def __init__(self, feed, show_video=True):
        pygame.init()
        pygame.display.set_caption("Lava Climb")
        self.feed       = feed
        self.show_video = show_video
        self.font  = pygame.font.SysFont("menlo,consolas,monospace", 16)
        self.big   = pygame.font.SysFont("menlo,consolas,monospace", 34, bold=True)
        self.roi_w, self.roi_h = 320, 640
        self.scale = 1.0
        self.screen = None
        self._fit_window()
        self.player = Player(self.lamp_w / 2, self.base_y)
        self.clock  = pygame.time.Clock()
        self.run_t  = 0.0
        self.best   = None
        self.state  = "play"      # play | win
        self.win_t  = 0.0
        self.deaths = 0
        self.plat_mask = None
        self.blob_draw = []       # (colour, screen_poly, vy_screen)
        self.win_flash = 0.0

    # window sized to the ROI aspect so the lamp fills it
    def _fit_window(self):
        self.scale  = WIN_H / float(self.roi_h)
        self.lamp_w = int(self.roi_w * self.scale)
        self.lamp_h = WIN_H
        self.base_y = self.lamp_h - BASE_H
        self.goal_y = int(self.lamp_h * GOAL_FRAC)
        w = self.lamp_w + 220                      # side panel
        self.screen = pygame.display.set_mode((w, self.lamp_h))
        self.plat_surf = pygame.Surface((self.lamp_w, self.lamp_h), pygame.SRCALPHA)

    def _maybe_resize(self):
        pkt = self.feed.packet
        if pkt and (pkt["w"], pkt["h"]) != (self.roi_w, self.roi_h):
            self.roi_w, self.roi_h = pkt["w"], pkt["h"]
            self._fit_window()
            self.player.reset(self.lamp_w / 2, self.base_y)

    # ── Platforms from packet ────────────────────────────────────────────
    def _build_platforms(self):
        """Rasterise extrapolated blob polygons into a collision mask."""
        self.plat_surf.fill((0, 0, 0, 0))
        self.blob_draw = []
        pkt = self.feed.packet
        if not pkt or not self.feed.alive:
            self.plat_mask = pygame.mask.from_surface(self.plat_surf)
            return
        dt_pkt = max(0.0, min(time.time() - self.feed.recv_t, 0.25))
        fps    = float(pkt.get("fps", 15.0))
        s      = self.scale
        for b in pkt["blobs"]:
            # velocity is px/frame in ROI space -> px/s screen
            dx = b["vx"] * fps * dt_pkt
            dy = b["vy"] * fps * dt_pkt
            poly = [((px + dx) * s, (py + dy) * s) for px, py in b["poly"]]
            if len(poly) < 3:
                continue
            pygame.draw.polygon(self.plat_surf, (255, 255, 255, 255), poly)
            col = COL_BLOB.get(b["dir"], COL_BLOB["turning"])
            self.blob_draw.append((col, poly, b["vy"] * fps * s, b["vx"] * fps * s))
        self.plat_mask = pygame.mask.from_surface(self.plat_surf)

    def _solid(self, x, y):
        if x < 0 or y < 0 or x >= self.lamp_w or y >= self.lamp_h:
            return False
        return self.plat_mask.get_at((int(x), int(y))) != 0

    def _blob_velocity_at(self, x, y):
        """Velocity of the blob under a point (for riding)."""
        pt = (x, y)
        for col, poly, vy, vx in self.blob_draw:
            if _point_in_poly(pt, poly):
                return vx, vy
        return 0.0, 0.0

    # ── Physics ──────────────────────────────────────────────────────────
    def _update_player(self, dt, keys):
        p = self.player
        left  = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        jump  = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]

        target = (right - left) * MOVE_SPEED
        accel  = 1.0 if p.grounded else AIR_CONTROL
        p.vx  += (target - p.vx) * min(1.0, 14 * accel * dt)
        if target:
            p.facing = 1 if target > 0 else -1

        # jump
        if jump and (p.grounded or p.coyote > 0) and not p.jumping:
            p.vy = JUMP_V + min(0.0, p.carrier_vy * 0.6)   # rising blob boosts
            p.grounded = False
            p.coyote = 0.0
            p.jumping = True
        if not jump:
            if p.jumping and p.vy < 0:
                p.vy *= JUMP_CUT
            p.jumping = False

        if not p.grounded:
            p.vy += GRAVITY * dt
            p.coyote = max(0.0, p.coyote - dt)

        # horizontal move + carrier drift
        p.x += (p.vx + (p.carrier_vx if p.grounded else 0)) * dt
        p.x  = min(max(p.x, PLAYER_W / 2 + 2), self.lamp_w - PLAYER_W / 2 - 2)

        # vertical move
        old_y = p.y
        if p.grounded:
            p.y += p.carrier_vy * dt
        else:
            p.y += p.vy * dt

        # ── landing / standing on blobs (one-way) ──
        feet_xs = (p.x - PLAYER_W * 0.35, p.x, p.x + PLAYER_W * 0.35)
        was_grounded = p.grounded
        p.grounded = False

        if p.vy >= 0 or was_grounded:
            # find surface: step from old_y down to new y, looking for first
            # solid pixel under any foot; snap to its top edge.
            landed = None
            y0, y1 = int(min(old_y, p.y)), int(max(old_y, p.y)) + 1
            for yy in range(y0, y1 + 1):
                if any(self._solid(fx, yy) for fx in feet_xs):
                    landed = yy
                    break
            if landed is None and was_grounded:
                # sunk into a blob that swelled upward: pop up a little
                for k in range(1, LAND_SNAP + 1):
                    yy = int(p.y) - k
                    if not any(self._solid(fx, yy) for fx in feet_xs):
                        if any(self._solid(fx, yy + 1) for fx in feet_xs):
                            landed = yy + 1
                        break
            if landed is not None:
                # walk up to the top edge of the solid run
                top = landed
                for k in range(LAND_SNAP):
                    if any(self._solid(fx, top - 1) for fx in feet_xs):
                        top -= 1
                    else:
                        break
                p.y = float(top)
                p.vy = 0.0
                p.grounded = True
                p.coyote = COYOTE_TIME
                p.carrier_vx, p.carrier_vy = self._blob_velocity_at(p.x, p.y + 2)

        # base platform
        if not p.grounded and p.vy >= 0 and p.y >= self.base_y:
            p.y = float(self.base_y)
            p.vy = 0.0
            p.grounded = True
            p.coyote = COYOTE_TIME
            p.carrier_vx = p.carrier_vy = 0.0

        if not p.grounded:
            p.carrier_vx = p.carrier_vy = 0.0

        # fell off the bottom (e.g. carried down past the base by a sinking blob)
        if p.y > self.lamp_h + 10:
            self.deaths += 1
            p.reset(self.lamp_w / 2, self.base_y)

        # goal
        if p.y - PLAYER_H <= self.goal_y:
            self.state = "win"
            self.win_t = self.run_t
            if self.best is None or self.run_t < self.best:
                self.best = self.run_t
            self.win_flash = 1.0

    # ── Drawing ──────────────────────────────────────────────────────────
    def _draw_lamp(self):
        sc = self.screen
        lamp = pygame.Rect(0, 0, self.lamp_w, self.lamp_h)
        sc.fill(COL_BG)
        if self.show_video and self.feed.video is not None:
            vid = pygame.transform.smoothscale(self.feed.video, (self.lamp_w, self.lamp_h))
            sc.blit(vid, (0, 0))
            dim = pygame.Surface((self.lamp_w, self.lamp_h), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 70))
            sc.blit(dim, (0, 0))
        else:
            # liquid gradient
            for i in range(0, self.lamp_h, 4):
                k = i / self.lamp_h
                col = (int(24 + 30 * k), int(14 + 8 * k), int(40 + 50 * (1 - k)))
                pygame.draw.rect(sc, col, (0, i, self.lamp_w, 4))

        # blobs
        ov = pygame.Surface((self.lamp_w, self.lamp_h), pygame.SRCALPHA)
        for col, poly, vy, vx in self.blob_draw:
            pygame.draw.polygon(ov, (*col, 110), poly)
            pygame.draw.polygon(ov, (*col, 230), poly, 2)
            # highlight the landable top edge
            top_y = min(p[1] for p in poly)
            xs = [p[0] for p in poly if p[1] < top_y + 6]
            if len(xs) >= 2:
                pygame.draw.line(ov, (255, 255, 255, 180),
                                 (min(xs), top_y), (max(xs), top_y), 2)
        sc.blit(ov, (0, 0))

        # goal zone + base
        gz = pygame.Surface((self.lamp_w, self.goal_y), pygame.SRCALPHA)
        pulse = 40 + int(30 * math.sin(time.time() * 3))
        gz.fill((*COL_GOAL, pulse))
        sc.blit(gz, (0, 0))
        pygame.draw.line(sc, COL_GOAL, (0, self.goal_y), (self.lamp_w, self.goal_y), 2)
        pygame.draw.rect(sc, COL_BASE, (0, self.base_y, self.lamp_w, BASE_H))
        pygame.draw.rect(sc, COL_GLASS, lamp, 3)

        self._draw_player()

        if not self.feed.alive:
            t = self.big.render("NO SIGNAL", True, (255, 80, 80))
            sc.blit(t, (self.lamp_w // 2 - t.get_width() // 2, self.lamp_h // 2 - 20))
            t2 = self.font.render("start audible_lava.py with LavaBridge", True, COL_DIM)
            sc.blit(t2, (self.lamp_w // 2 - t2.get_width() // 2, self.lamp_h // 2 + 20))

    def _draw_player(self):
        p = self.player
        sc = self.screen
        x, y = int(p.x), int(p.y)
        bob = int(2 * math.sin(time.time() * 10)) if (p.grounded and abs(p.vx) > 30) else 0
        body = (70, 130, 230)
        skin = (90, 150, 240)
        white = (245, 245, 245)
        # legs
        pygame.draw.line(sc, body, (x - 4, y - 9 + bob), (x - 5, y), 3)
        pygame.draw.line(sc, body, (x + 4, y - 9 - bob), (x + 5, y), 3)
        pygame.draw.ellipse(sc, white, (x - 8, y - 3, 7, 4))
        pygame.draw.ellipse(sc, white, (x + 1, y - 3, 7, 4))
        # trousers + torso
        pygame.draw.rect(sc, white, (x - 6, y - 13, 12, 6), border_radius=2)
        pygame.draw.rect(sc, body, (x - 5, y - 19, 10, 7), border_radius=2)
        # arms
        ax = p.facing
        pygame.draw.line(sc, body, (x - 5 * ax, y - 17), (x - 9 * ax, y - 12), 3)
        pygame.draw.line(sc, body, (x + 5 * ax, y - 17), (x + 9 * ax, y - 21), 3)
        # head
        pygame.draw.circle(sc, skin, (x, y - 24), 6)
        pygame.draw.circle(sc, (20, 20, 30), (x + 2 * ax, y - 25), 1)
        # floppy white hat
        pts = [(x - 6, y - 27), (x + 6, y - 27), (x + 2, y - 36),
               (x - 4 - 2 * ax, y - 38 + (1 if ax > 0 else -1)), (x - 8, y - 31)]
        pygame.draw.polygon(sc, white, pts)

    def _draw_panel(self):
        sc = self.screen
        x0 = self.lamp_w + 16
        y = 18
        def line(txt, col=COL_TEXT, dy=22, f=None):
            nonlocal y
            sc.blit((f or self.font).render(txt, True, col), (x0, y))
            y += dy
        line("LAVA CLIMB", COL_GOAL, 34, self.big)
        line(f"time   {self.run_t:6.1f}s")
        line(f"best   {'--' if self.best is None else f'{self.best:.1f}s'}")
        line(f"falls  {self.deaths}")
        h = 1.0 - (self.player.y - PLAYER_H) / self.lamp_h
        line(f"height {100 * max(0, h):5.1f}%")
        y += 8
        n = len(self.feed.packet["blobs"]) if self.feed.packet else 0
        line(f"blobs  {n}", COL_DIM)
        line(f"feed   {'live' if self.feed.alive else 'stale'}", COL_DIM)
        line(f"pkts   {self.feed.count}", COL_DIM)
        y += 14
        for name, col in COL_BLOB.items():
            pygame.draw.rect(sc, col, (x0, y + 4, 12, 12))
            sc.blit(self.font.render(name, True, COL_DIM), (x0 + 18, y))
            y += 20
        y += 14
        for t in ("A/D  move", "SPACE jump", "R  restart", "ESC quit"):
            line(t, COL_DIM, 20)

        if self.state == "win":
            box = pygame.Rect(0, self.lamp_h // 2 - 60, self.lamp_w, 120)
            s = pygame.Surface(box.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 190))
            sc.blit(s, box.topleft)
            t = self.big.render("TOP!", True, COL_GOAL)
            sc.blit(t, (box.centerx - t.get_width() // 2, box.y + 18))
            t2 = self.font.render(f"{self.win_t:.1f}s  --  press R", True, COL_TEXT)
            sc.blit(t2, (box.centerx - t2.get_width() // 2, box.y + 70))

    # ── Main loop ────────────────────────────────────────────────────────
    def restart(self):
        self.player.reset(self.lamp_w / 2, self.base_y)
        self.run_t = 0.0
        self.deaths = 0
        self.state = "play"

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 1 / 30)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_r:
                        self.restart()

            if isinstance(self.feed, SimFeed):
                self.feed.step(dt)
            else:
                self.feed.poll()
            self._maybe_resize()
            self._build_platforms()

            if self.state == "play":
                self.run_t += dt
                self._update_player(dt, pygame.key.get_pressed())

            self._draw_lamp()
            self._draw_panel()
            pygame.display.flip()
        pygame.quit()


def _point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def main():
    args = sys.argv[1:]
    port = PORT
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    show_video = "--no-video" not in args
    if "--sim" in args:
        feed = SimFeed()
        print("[LavaClimb] synthetic lamp")
    else:
        feed = BlobFeed(port)
        print(f"[LavaClimb] listening on udp://127.0.0.1:{port}")
    Game(feed, show_video=show_video).run()


if __name__ == "__main__":
    main()
