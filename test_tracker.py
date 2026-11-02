#!/usr/bin/env python3
# test_tracker.py -- Audible Lava v3.1
# Live visual test of flow-advected tracking.
# Colour-codes blobs by direction. Shows coast_frames counter.
# Prints all four lifecycle event types.
#
# v3.1 fix:
#   The previous version unpacked tracker.events as
#       for ev_type, ev_id in tracker.events:
#   which crashes the moment a 3-tuple ('merge', merged, survivor) or
#   ('split', parent, child) event fires. Events are now unpacked by
#   index so all four event shapes are handled. The same defensive
#   pattern is used in audible_lava.py.

import cv2
import numpy as np
import random

from lamp_profile   import LampProfile, PROFILE_PATH
from blob_detection import BlobDetector
from blob_tracker   import BlobTracker
from blob_detection import RISING, FALLING, TURNING

DIR_COLOURS = {RISING: (80,255,128), FALLING: (80,128,255), TURNING: (255,220,80)}
CAMERA_INDEX = 0


def _format_event(event, tracked):
    """
    Format a tracker event for one-line console output.

    Handles all four event shapes:
        ('birth', id)
        ('death', id)
        ('merge', merged_id, survivor_id)
        ('split', parent_id, child_id)
    """
    ev_type = event[0]
    primary_id = event[1] if len(event) >= 2 else -1
    other_id   = event[2] if len(event) >= 3 else None

    # Find the alive blob (if any) to attach a direction tag
    blob    = next((b for b in tracked if b.id == primary_id), None)
    dir_tag = f" [{blob.direction}]" if blob else ""

    if ev_type == "birth":
        return f"  + birth: ID {primary_id}{dir_tag}"
    if ev_type == "death":
        return f"  - death: ID {primary_id}{dir_tag}"
    if ev_type == "merge":
        # primary_id is the merged (retired) blob, other_id is the survivor
        return f"  > merge: ID {primary_id} → ID {other_id}"
    if ev_type == "split":
        # primary_id is parent (kept), other_id is child (new)
        return f"  < split: ID {primary_id} → child ID {other_id}{dir_tag}"
    return f"  ? {ev_type}: {event[1:]}"


def main():
    profile  = LampProfile.load(PROFILE_PATH) or LampProfile.default()
    detector = BlobDetector(profile=profile)
    cap      = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Camera")
        return

    profile.apply_to_capture(cap)

    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    profile.frame_h = h
    profile.frame_w = w

    tracker = BlobTracker(profile=profile)

    id_colours = {}
    def get_col(bid):
        if bid not in id_colours:
            rng = random.Random(bid * 137 + 42)
            id_colours[bid] = (
                rng.randint(80, 255),
                rng.randint(80, 255),
                rng.randint(80, 255),
            )
        return id_colours[bid]

    print("Tracker test: q=quit")
    print("  Green ring = rising  |  Blue ring = falling  |  Yellow = turning")
    print("  Events: + birth   - death   > merge   < split")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # v4.1: detector/tracker run on the lamp ROI; build the overlay on the
        # ROI too so ROI-sized masks and ROI-local centroids line up.
        roi  = profile.crop(frame)
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blobs   = detector.detect(roi, grey=grey)
        tracked = tracker.update(blobs, roi, grey=grey)

        # Defensive event unpacking — handles 2- and 3-tuples uniformly.
        for event in tracker.events:
            print(_format_event(event, tracked))

        overlay = roi.copy()

        # Draw advected masks as faint overlays
        for tid, track in tracker._tracks.items():
            mask = track["mask"]
            if mask.sum() == 0:
                continue
            col = get_col(tid)
            overlay[mask > 0] = (
                overlay[mask > 0].astype(np.float32) * 0.7
                + np.array(col, dtype=np.float32) * 0.3
            ).clip(0, 255).astype(np.uint8)

        for b in tracked:
            cx, cy  = int(b.centroid[0]), int(b.centroid[1])
            col     = get_col(b.id)
            dir_col = DIR_COLOURS.get(b.direction, (255, 255, 255))
            cv2.circle(overlay, (cx, cy), 12, col,     2)
            cv2.circle(overlay, (cx, cy),  8, dir_col, -1)
            cv2.putText(overlay, f"ID:{b.id}",
                        (cx + 14, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
            cv2.putText(overlay,
                f"age={b.age} coast={b.coast_frames} {b.direction[0].upper()}",
                (cx + 14, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)

        cv2.putText(overlay, f"Tracks: {len(tracked)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.imshow("Tracker v4.1 (ROI)", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
