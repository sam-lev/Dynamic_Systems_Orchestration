# lava_bridge.py -- publish tracked blob geometry to a game process over UDP
#
# Audible Lava runs at TARGET_FPS (15) with one pygame window already open
# (the keyboard controller), so the game lives in a second process
# (lava_climb.py) and receives blob state here.  One JSON datagram per
# frame on localhost:9100:
#
#   {"t": <unix time>, "w": roi_w, "h": roi_h, "fps": synth fps,
#    "blobs": [{"id", "cx", "cy", "vx", "vy", "dir", "poly": [[x,y],..]}],
#    "jpg": "<base64 jpeg of the ROI>"   # optional live video background
#   }
#
# Coordinates are ROI pixels (same frame the tracker uses); the game
# rescales.  Polygons come from each track's advected mask, so a blob's
# platform outline survives between detection cycles and across the
# transparency fluctuations the synth side already handles.
#
# Hook-up in audible_lava.py (three lines):
#     from lava_bridge import LavaBridge
#     bridge = LavaBridge(profile, fps=TARGET_FPS)        # after tracker init
#     bridge.publish(tracker, roi)                         # after tracker.update(...)

import base64
import json
import socket
import time

import cv2
import numpy as np

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9100
MAX_DGRAM   = 60000          # stay under the 64 KiB UDP ceiling


class LavaBridge:
    def __init__(self, profile, host=BRIDGE_HOST, port=BRIDGE_PORT,
                 fps=15, send_video=True, video_width=240, jpeg_quality=55,
                 poly_epsilon=2.0):
        self.profile      = profile
        self.addr         = (host, port)
        self.fps          = fps
        self.send_video   = send_video
        self.video_width  = video_width
        self.jpeg_quality = jpeg_quality
        self.poly_epsilon = poly_epsilon
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_err = 0.0

    # ── Geometry ──────────────────────────────────────────────────────────
    def _polygons_from_mask(self, mask):
        """Simplified outer contours of a track mask, in ROI px."""
        if mask is None or not mask.any():
            return []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            if len(c) < 3 or cv2.contourArea(c) < 30:
                continue
            c = cv2.approxPolyDP(c, self.poly_epsilon, True)
            if len(c) >= 3:
                polys.append(c.reshape(-1, 2).astype(int).tolist())
        return polys

    # ── Publish ───────────────────────────────────────────────────────────
    def publish(self, tracker, roi=None):
        blobs = []
        for tid, tr in tracker._tracks.items():
            b = tr["blob"]
            if not b.alive:
                continue
            for poly in self._polygons_from_mask(tr.get("mask")):
                blobs.append({
                    "id":   int(b.id),
                    "cx":   float(b.centroid[0]),
                    "cy":   float(b.centroid[1]),
                    "vx":   float(b.vx),
                    "vy":   float(b.vy),
                    "dir":  str(b.direction),
                    "poly": poly,
                })

        msg = {
            "t":     time.time(),
            "w":     int(self.profile.roi_w),
            "h":     int(self.profile.roi_h),
            "fps":   self.fps,
            "blobs": blobs,
        }

        if self.send_video and roi is not None:
            h, w = roi.shape[:2]
            scale = self.video_width / float(w)
            small = cv2.resize(roi, (self.video_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                msg["jpg"] = base64.b64encode(buf.tobytes()).decode("ascii")

        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_DGRAM and "jpg" in msg:
            del msg["jpg"]                       # drop video before geometry
            data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_DGRAM:
            # Still too big: thin polygons.
            for b in msg["blobs"]:
                b["poly"] = b["poly"][::2]
            data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        try:
            self.sock.sendto(data, self.addr)
        except OSError as e:
            now = time.time()
            if now - self._last_err > 5:
                print(f"[LavaBridge] send failed: {e}")
                self._last_err = now

    def close(self):
        self.sock.close()
