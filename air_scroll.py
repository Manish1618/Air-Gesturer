#!/usr/bin/env python3
"""Camera based air gestures for Linux/Wayland (Hyprland).

Gestures (all directions as seen in the mirrored preview):
  point + move      scroll up/down (and left/right)
  open palm swipe   left/right/up/down actions (default: switch workspace)
  two fingers + move  step actions (default: volume up/down)
  hold a fist       pause / resume all gestures
Send SIGUSR1 (pkill -USR1 -f air_scroll.py) to toggle pause from a keybind.
"""

from __future__ import annotations

import argparse
import signal
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

# Catch pause toggles before the slow imports below; SIGUSR1's default
# action would otherwise kill the process while it is still starting up.
_early_toggle = {"pending": False}
signal.signal(signal.SIGUSR1, lambda *_: _early_toggle.update(pending=True))

import cv2
import mediapipe as mp

import gestures as g
from actions import Actions

HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"Config {path} not found; using defaults.", file=sys.stderr)
        return {}


def open_camera(index: int, fixed_framerate: bool = True):
    if fixed_framerate and shutil.which("v4l2-ctl"):
        # Many webcams halve their frame rate in dim light to expose longer,
        # which doubles gesture latency. Keep the full rate instead.
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}",
                        "--set-ctrl=exposure_dynamic_framerate=0"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Keep the driver's default buffers: a 1-frame buffer makes the camera drop
    # frames (~27 fps), and processing is fast enough that frames never queue.
    return camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--no-preview", action="store_true", help="run without a preview window")
    parser.add_argument("--dry-run", action="store_true", help="print actions instead of sending them")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc, sw = cfg.get("scroll", {}), cfg.get("swipe", {})
    acts_cfg = cfg.get("actions", {})
    cam_cfg = cfg.get("camera", {})

    actions = Actions(invert_scroll=sc.get("invert", False), dry_run=args.dry_run)
    if not actions.can_scroll and not args.dry_run:
        print("Warning: ydotool not found; scrolling disabled, other gestures still work.",
              file=sys.stderr)

    # Retry for a while: a previous instance may still be releasing the camera.
    for attempt in range(10):
        camera = open_camera(args.camera, cam_cfg.get("fixed_framerate", True))
        if camera.isOpened():
            break
        camera.release()
        time.sleep(1)
    else:
        print(f"Could not open camera {args.camera}. Check camera access or try --camera 1.",
              file=sys.stderr)
        return 2

    hands_api = mp.solutions.hands
    tracker = hands_api.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                              min_detection_confidence=0.65, min_tracking_confidence=0.6)

    pose_filter = g.PoseFilter(frames=cfg.get("tracking", {}).get("confirm_frames", 2))
    smoother = g.Smoother(alpha=0.5)
    scroller = g.Scroller(step=sc.get("step", 0.08), deadzone=sc.get("deadzone", 0.006),
                          horizontal=sc.get("horizontal", True), max_ticks=sc.get("max_ticks", 6))
    swiper = g.SwipeDetector(distance=sw.get("distance", 1.3), window=sw.get("window", 0.35),
                             cooldown=sw.get("cooldown", 0.8))
    stepper = g.Stepper(step=cfg.get("two_finger", {}).get("step", 0.35))
    fist_hold = g.HoldDetector(hold=cfg.get("pause", {}).get("hold", 1.0))
    ok_hold = g.HoldDetector(hold=cfg.get("ok_sign", {}).get("hold", 0.8))

    state = {"paused": args.start_paused, "toggle": _early_toggle["pending"]}
    signal.signal(signal.SIGUSR1, lambda *_: state.update(toggle=True))
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    def set_paused(paused: bool) -> None:
        state["paused"] = paused
        actions.notify("Paused" if paused else "Resumed")
        print("Paused." if paused else "Resumed.")

    prev_pose = g.NONE
    last_action, last_action_t = "", 0.0
    print("Air Scroll running. Point to scroll, swipe an open palm, hold a fist to pause. Q quits.")
    read_failures = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                # Signals (pause toggles) and camera hiccups make single reads
                # fail; retry, and reopen the camera if it keeps failing.
                read_failures += 1
                if read_failures == 30:
                    print("Camera frame read failed; reopening camera.", file=sys.stderr)
                if read_failures % 30 == 0:
                    camera.release()
                    time.sleep(1)
                    camera = open_camera(args.camera, cam_cfg.get("fixed_framerate", True))
                else:
                    time.sleep(0.03)
                continue
            if read_failures >= 30:
                print("Camera recovered.", file=sys.stderr)
            read_failures = 0
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if state["toggle"]:
                state["toggle"] = False
                set_paused(not state["paused"])

            hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
            pose = pose_filter.update(g.classify(hand.landmark) if hand else g.NONE)

            if pose != prev_pose:
                smoother.reset(); scroller.reset(); swiper.reset(); stepper.reset()
                prev_pose = pose

            if fist_hold.hold > 0 and fist_hold.update(pose == g.FIST, now):
                set_paused(not state["paused"])

            fired = ""
            if ok_hold.update(pose == g.OK and not state["paused"], now):
                actions.run(acts_cfg.get("ok_sign", ""))
                fired = "ok sign"
            if hand and not state["paused"]:
                lm = hand.landmark
                palm = g.palm_size(lm)
                if pose == g.POINT:
                    x, y = smoother.update(lm[8].x, lm[8].y)
                    right, down = scroller.update(x, y, palm)
                    name = "point_down" if down > 0 else "point_up"
                    if down and acts_cfg.get(name):
                        # A configured action (e.g. an arrow key) replaces wheel ticks.
                        for _ in range(min(abs(down), 3)):
                            actions.run(acts_cfg[name])
                        fired = name.replace("_", " ")
                    elif right or down:
                        if actions.scroll(right, down):
                            fired = ("scroll " + ("down" if down > 0 else "up") if down
                                     else "scroll " + ("right" if right > 0 else "left"))
                elif pose == g.PALM:
                    x, y = smoother.update(lm[9].x, lm[9].y)
                    direction = swiper.update(x, y, palm, now)
                    if direction:
                        actions.run(acts_cfg.get(f"swipe_{direction}", ""))
                        fired = f"swipe {direction}"
                elif pose == g.TWO:
                    x, y = smoother.update(lm[12].x, lm[12].y)
                    steps = stepper.update(y, palm)
                    name = "two_down" if steps > 0 else "two_up"
                    for _ in range(min(abs(steps), 3)):
                        actions.run(acts_cfg.get(name, ""))
                    if steps:
                        fired = name.replace("_", " ")
            if fired:
                last_action, last_action_t = fired, now

            if not args.no_preview:
                if hand:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                draw_hud(frame, pose, state["paused"], actions.scroll_error,
                         last_action if now - last_action_t < 1.0 else "",
                         max(fist_hold.progress(now), ok_hold.progress(now)))
                cv2.imshow("Air Scroll", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


HINTS = {
    g.NONE: "Show your hand",
    g.POINT: "Point: move to scroll",
    g.PALM: "Palm: swipe for actions",
    g.TWO: "Two fingers: move up/down",
    g.FIST: "Fist: hold to pause/resume",
    g.OK: "OK sign: hold to close window",
    g.OTHER: "Point / palm / two fingers / fist",
}


def draw_hud(frame, pose, paused, error, action, hold_progress) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (60, 160, 255) if paused else (40, 230, 120)
    title = "PAUSED - hold fist to resume" if paused else HINTS.get(pose, "")
    cv2.putText(frame, title, (16, 32), font, 0.65, color, 2, cv2.LINE_AA)
    if action:
        cv2.putText(frame, action, (16, 64), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if error:
        cv2.putText(frame, error[:70], (16, frame.shape[0] - 40), font, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
    if hold_progress > 0:
        w = int((frame.shape[1] - 32) * hold_progress)
        cv2.rectangle(frame, (16, 76), (16 + w, 84), color, -1)
    cv2.putText(frame, "Q or Ctrl+C in terminal: quit", (16, frame.shape[0] - 16), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
