#!/usr/bin/env python3
"""
Air Gesture Controller - a touch-free desktop controller using the webcam.

Control your computer with hand gestures in the air:
  * Point (index finger) and move up/down  -> scroll / arrow keys
  * Open palm quick swipe left/right        -> switch workspace
  * Two fingers up, move up/down            -> volume up / down
  * OK sign, hold 0.8 s                     -> close active window
  * Fist, hold 4 s                          -> pause / resume gestures

The app has a desktop GUI (Tkinter) that shows the live camera with the
detected hand skeleton, the current gesture, FPS, an activity log and usage
statistics. All camera processing happens locally; no frames are saved or
sent anywhere (except a screenshot when you press the Screenshot button).

Modules used:
  gestures.py - hand pose classification + gesture state machines
  actions.py  - sends scroll/keys/commands to the desktop
  config.toml - sensitivity and gesture -> action mapping

Run:  .venv/bin/python air_gesture_app.py
"""

from __future__ import annotations

import json
import os
import queue
import sys
import shutil
import subprocess
import threading
import time
import tkinter as tk
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp

import gestures as g
from actions import Actions

APP_NAME = "Air Gesture Controller"
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"
if "--config" in sys.argv[:-1]:
    CONFIG_PATH = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
SHOTS_DIR = HERE / "screenshots"
LOG_DIR = HERE / "logs"

FONT = "Noto Sans"

# ---- colours for the dark theme -------------------------------------------
BG, PANEL, CARD = "#14161c", "#1c1f27", "#252935"
FG, MUTED, ACCENT = "#e8eaf0", "#8a90a2", "#3ddc84"
WARN, DANGER, BLUE = "#ffb347", "#ff5c5c", "#5aa9ff"

# Gesture reference shown in the GUI: (pose, gesture, action)
GESTURE_GUIDE = [
    ("point", "Index finger + move", "Scroll up / down"),
    ("palm", "Open palm quick swipe", "Switch workspace"),
    ("two", "Two fingers + move", "Volume up / down"),
    ("ok", "OK sign, hold 0.8 s", "Close window"),
    ("fist", "Fist, hold 4 s", "Pause / resume"),
]

# Guided screenshot tour: (file name, on-screen instruction)
TOUR_STEPS = [
    ("camera_running", "Keep your hand OUT of the camera view for a moment"),
    ("pointing", "Raise only your INDEX finger and move it slowly up and down"),
    ("open_palm", "Show an OPEN PALM, then swipe it quickly left or right"),
    ("two_fingers", "Raise INDEX + MIDDLE fingers and move your hand up / down"),
    ("ok_sign", "Make an OK sign (thumb tip touches index tip) and hold it"),
    ("fist_paused", "Make a FIST and hold it for 4 seconds to PAUSE"),
    ("resumed", "Hold a FIST again for 4 seconds to RESUME"),
    ("session_summary", "All done - lower your hand"),
]
# Demo-video script (--video): gestures really control the window on the right.
VIDEO_STEPS = [
    ("intro", "Air Gesture Controller demo - show your hand to the camera"),
    ("scroll_down", "POINT your index finger and move it DOWN: the document scrolls down"),
    ("scroll_up", "Now move your finger UP: the document scrolls back up"),
    ("swipe", "OPEN PALM: swipe quickly LEFT for the next page"),
    ("volume", "TWO FINGERS up/down: system volume changes"),
    ("ok_sign", "Make an OK sign and hold it (detected; set to no action in this demo)"),
    ("fist_paused", "Hold a FIST for 4 seconds to PAUSE gesture control"),
    ("resumed", "Hold the FIST again for 4 seconds to RESUME"),
    ("finish", "Demo complete - thank you!"),
]
STEP_TIMEOUT = 45.0     # seconds; take the screenshot anyway after this

POSE_LABELS = {
    g.NONE: "No hand", g.POINT: "Pointing", g.PALM: "Open palm",
    g.TWO: "Two fingers", g.FIST: "Fist", g.OK: "OK sign", g.OTHER: "Unknown pose",
}


def load_config(path: Path) -> dict:
    """Read config.toml; fall back to built-in defaults if it is missing."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a webcam at 640x480 (MJPG for a steady 30 fps)."""
    if shutil.which("v4l2-ctl"):
        # Keep full frame rate in dim light (lower gesture latency).
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}",
                        "--set-ctrl=exposure_dynamic_framerate=0"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


# ============================================================================
#  Gesture engine (runs in a background thread)
# ============================================================================
class GestureEngine(threading.Thread):
    """Reads camera frames, detects hand gestures and fires desktop actions.

    It talks to the GUI only through `events` (a thread-safe queue), so the
    window stays responsive while frames are being processed.
    """

    def __init__(self, camera_index: int, cfg: dict, events: queue.Queue,
                 demo_mode: bool, scroll_step: float):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cfg = cfg
        self.events = events
        self.demo_mode = demo_mode          # True = log actions only
        self.paused = False
        self._stop_evt = threading.Event()
        self._toggle_pause = threading.Event()

        sc, sw = cfg.get("scroll", {}), cfg.get("swipe", {})
        self.acts = cfg.get("actions", {})
        self.live_actions = Actions(invert_scroll=sc.get("invert", False), dry_run=False)

        # Gesture state machines from gestures.py
        self.pose_filter = g.PoseFilter(frames=cfg.get("tracking", {}).get("confirm_frames", 2))
        self.smoother = g.Smoother(alpha=0.5)
        self.scroller = g.Scroller(step=scroll_step, deadzone=sc.get("deadzone", 0.006),
                                   horizontal=sc.get("horizontal", True),
                                   max_ticks=sc.get("max_ticks", 6))
        self.swiper = g.SwipeDetector(distance=sw.get("distance", 0.55),
                                      window=sw.get("window", 0.5),
                                      cooldown=sw.get("cooldown", 0.8))
        self.stepper = g.Stepper(step=cfg.get("two_finger", {}).get("step", 0.15))
        self.fist_hold = g.HoldDetector(hold=cfg.get("pause", {}).get("hold", 4.0))
        self.ok_hold = g.HoldDetector(hold=cfg.get("ok_sign", {}).get("hold", 0.8))

    # ---- control from the GUI thread -------------------------------------
    def stop(self) -> None:
        self._stop_evt.set()

    def toggle_pause(self) -> None:
        self._toggle_pause.set()

    def set_scroll_step(self, step: float) -> None:
        self.scroller.step = step

    # ---- helpers ----------------------------------------------------------
    def _emit(self, kind: str, **data) -> None:
        self.events.put((kind, data))

    def _fire(self, gesture: str, spec: str) -> None:
        """Run a configured action (or only log it in demo mode)."""
        if spec and not spec.startswith("(") and not self.demo_mode:
            self.live_actions.run(spec)
        self._emit("action", gesture=gesture, spec=spec or "(not assigned)",
                   demo=self.demo_mode)

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        self._emit("paused", paused=paused)

    # ---- main loop ----------------------------------------------------------
    def run(self) -> None:
        cam = None
        for _ in range(5):                       # camera may still be busy
            cam = open_camera(self.camera_index)
            if cam.isOpened():
                break
            cam.release()
            time.sleep(0.6)
        if cam is None or not cam.isOpened():
            self._emit("error", text=f"Could not open camera {self.camera_index}.")
            return

        hands = mp.solutions.hands
        tracker = hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                              min_detection_confidence=0.65, min_tracking_confidence=0.6)
        draw = mp.solutions.drawing_utils
        self._emit("started")

        prev_pose, fps, t_prev, fails = g.NONE, 0.0, time.monotonic(), 0
        try:
            while not self._stop_evt.is_set():
                ok, frame = cam.read()
                if not ok:
                    fails += 1
                    if fails > 60:
                        self._emit("error", text="Camera stopped sending frames.")
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                frame = cv2.flip(frame, 1)                       # mirror view
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now

                if self._toggle_pause.is_set():
                    self._toggle_pause.clear()
                    self._set_paused(not self.paused)

                result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
                pose = self.pose_filter.update(g.classify(hand.landmark) if hand else g.NONE)

                if pose != prev_pose:                            # new pose = new gesture
                    for sm in (self.smoother, self.scroller, self.swiper, self.stepper):
                        sm.reset()
                    prev_pose = pose

                if self.fist_hold.hold > 0 and self.fist_hold.update(pose == g.FIST, now):
                    self._set_paused(not self.paused)
                if self.ok_hold.update(pose == g.OK and not self.paused, now):
                    self._fire("OK sign", self.acts.get("ok_sign", ""))

                if hand and not self.paused:
                    self._handle_motion(pose, hand.landmark)

                # draw skeleton + hold progress bar on the preview
                if hand:
                    draw.draw_landmarks(frame, hand, hands.HAND_CONNECTIONS,
                                        draw.DrawingSpec((80, 220, 60), 2, 3),
                                        draw.DrawingSpec((255, 255, 255), 2))
                progress = max(self.fist_hold.progress(now), self.ok_hold.progress(now))
                if progress > 0:
                    w = int((frame.shape[1] - 32) * progress)
                    cv2.rectangle(frame, (16, frame.shape[0] - 24),
                                  (16 + w, frame.shape[0] - 14), (80, 200, 255), -1)

                ok_enc, ppm = cv2.imencode(".ppm", frame)
                if ok_enc:
                    self._emit("frame", ppm=ppm.tobytes(), pose=pose, fps=fps)
        finally:
            tracker.close()
            cam.release()
            self._emit("stopped")

    def _handle_motion(self, pose: str, lm) -> None:
        """Turn hand movement into scroll / swipe / volume gestures."""
        palm = g.palm_size(lm)
        now = time.monotonic()
        if pose == g.POINT:
            x, y = self.smoother.update(lm[8].x, lm[8].y)        # index fingertip
            right, down = self.scroller.update(x, y, palm)
            if down:
                name = "point_down" if down > 0 else "point_up"
                label = "Scroll down" if down > 0 else "Scroll up"
                spec = self.acts.get(name, "")
                if spec:
                    for _ in range(min(abs(down), 3)):
                        self._fire(label, spec)
                else:
                    if not self.demo_mode:
                        self.live_actions.scroll(0, down)
                    self._fire(label, "(scroll wheel)")
            elif right:
                if not self.demo_mode:
                    self.live_actions.scroll(right, 0)
                self._fire("Scroll right" if right > 0 else "Scroll left", "(scroll wheel)")
        elif pose == g.PALM:
            x, y = self.smoother.update(lm[9].x, lm[9].y)        # palm centre
            direction = self.swiper.update(x, y, palm, now)
            if direction:
                self._fire(f"Swipe {direction}", self.acts.get(f"swipe_{direction}", ""))
        elif pose == g.TWO:
            _, y = self.smoother.update(lm[12].x, lm[12].y)      # middle fingertip
            steps = self.stepper.update(y, palm)
            if steps:
                name = "two_down" if steps > 0 else "two_up"
                label = "Volume down" if steps > 0 else "Volume up"
                for _ in range(min(abs(steps), 3)):
                    self._fire(label, self.acts.get(name, ""))


# ============================================================================
#  GUI
# ============================================================================
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.configure(bg=BG)
        self.geometry("1240x760")
        self.minsize(1100, 700)

        self.cfg = load_config(CONFIG_PATH)
        self.events: queue.Queue = queue.Queue()
        self.engine: GestureEngine | None = None
        self.photo: tk.PhotoImage | None = None
        self.stats: Counter = Counter()
        self.log_lines: list[str] = []
        self.session_start: float | None = None
        self.current_pose = g.NONE
        self.pose_since = time.monotonic()
        self.last_action, self.last_action_t = "", 0.0
        self.tour_i: int | None = None          # index into TOUR_STEPS while touring
        tour_arg = next((a for a in sys.argv if a.startswith("--tour")), "")
        self.tour_auto = bool(tour_arg)          # start the tour on launch, quit after
        self.tour_first = int(tour_arg.partition("=")[2] or 1)   # --tour=4 starts at step 4
        self.tour_step_t0 = 0.0
        self.tour_actions = 0
        self.tour_events: dict[str, float] = {}
        self.tour_gestures: Counter = Counter()  # gestures seen in the current step
        self.video_mode = "--video" in sys.argv
        self.tour_steps = VIDEO_STEPS if self.video_mode else TOUR_STEPS
        if self.video_mode:
            self.tour_auto = True

        self.demo_var = tk.BooleanVar(value=False)
        self.camera_var = tk.IntVar(value=0)
        self.sens_var = tk.DoubleVar(value=self.cfg.get("scroll", {}).get("step", 0.08))

        self._style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<space>", lambda e: self.toggle_pause())
        self.bind("<F12>", lambda e: self.screenshot(0))
        self.after(20, self.poll_events)
        self.after(1000, self.tick_clock)
        self.after(400, self._hypr_float)
        if self.tour_auto:
            self.after(1500, self.start_tour)
        self.log("App started. Press 'Start Camera' to begin.", "info")

    # ---- look & feel --------------------------------------------------------
    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=FG, font=(FONT, 10))
        s.configure("TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=PANEL, foreground=FG)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=(FONT, 9))
        s.configure("Big.TLabel", background=CARD, foreground=FG, font=(FONT, 15, "bold"))
        s.configure("Head.TLabel", background=PANEL, foreground=FG, font=(FONT, 11, "bold"))
        s.configure("Title.TLabel", background=BG, foreground=FG, font=(FONT, 17, "bold"))
        s.configure("Sub.TLabel", background=BG, foreground=MUTED)
        s.configure("TButton", background=CARD, foreground=FG, borderwidth=0, padding=(12, 7))
        s.map("TButton", background=[("active", "#323849"), ("disabled", "#1f222b")],
              foreground=[("disabled", "#555")])
        s.configure("Accent.TButton", background=ACCENT, foreground="#08200f",
                    font=(FONT, 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#5cf0a0")])
        s.configure("Danger.TButton", background=DANGER, foreground="#2a0606",
                    font=(FONT, 10, "bold"))
        s.map("Danger.TButton", background=[("active", "#ff7f7f")])
        s.configure("TCheckbutton", background=PANEL, foreground=FG)
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=FG,
                    rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=MUTED, borderwidth=0)
        s.configure("TSpinbox", fieldbackground=CARD, foreground=FG, background=CARD,
                    arrowcolor=FG)
        s.configure("Horizontal.TScale", background=PANEL, troughcolor=CARD)

    def _card(self, parent, title: str) -> tuple[ttk.Frame, ttk.Label]:
        f = ttk.Frame(parent, style="Card.TFrame", padding=(12, 8))
        ttk.Label(f, text=title, style="Muted.TLabel").pack(anchor="w")
        value = ttk.Label(f, text="-", style="Big.TLabel")
        value.pack(anchor="w")
        return f, value

    def _build(self) -> None:
        # header
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 6))
        ttk.Label(head, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="Touch-free computer control with hand gestures",
                  style="Sub.TLabel").pack(side="left", padx=14, pady=(6, 0))
        self.clock = ttk.Label(head, text="", style="Sub.TLabel")
        self.clock.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # ---- left: camera preview ----
        cam_frame = ttk.Frame(body, padding=8)
        cam_frame.grid(row=0, column=0, sticky="n")
        # a blank 640x480 image makes the label size itself in pixels, not characters
        self.blank = tk.PhotoImage(width=640, height=480)
        self.video = tk.Label(cam_frame, bg="#0b0c10", image=self.blank, bd=0,
                              fg=MUTED, font=(FONT, 13), compound="center",
                              text="Camera is off\n\nPress  ▶ Start Camera")
        self.video.pack()
        self.banner = tk.Label(cam_frame, text="", bg=PANEL, fg=FG,
                               font=(FONT, 13, "bold"), anchor="w", padx=6)
        self.banner.pack(fill="x", pady=(6, 0))
        self.tour_label = tk.Label(cam_frame, text="", bg="#3a3216", fg="#ffd866",
                                   font=(FONT, 12, "bold"), anchor="w", padx=8, pady=5,
                                   wraplength=620, justify="left")

        # ---- left bottom: activity log ----
        log_frame = ttk.Frame(body, padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        top = ttk.Frame(log_frame)
        top.pack(fill="x")
        ttk.Label(top, text="Activity log", style="Head.TLabel").pack(side="left")
        ttk.Button(top, text="Export log", command=self.export_log).pack(side="right")
        ttk.Button(top, text="Clear", command=self.clear_log).pack(side="right", padx=6)
        self.log_box = tk.Text(log_frame, height=7, bg=CARD, fg=FG, bd=0,
                               font=("Noto Sans Mono", 9), padx=8, pady=6,
                               insertbackground=FG, state="disabled", wrap="none")
        self.log_box.pack(fill="both", expand=True, pady=(6, 0))
        for tag, col in (("info", MUTED), ("action", ACCENT), ("demo", BLUE),
                         ("warn", WARN), ("error", DANGER)):
            self.log_box.tag_configure(tag, foreground=col)

        # ---- right: status, controls, guide, stats ----
        side = ttk.Frame(body, padding=(12, 8))
        side.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(14, 0))

        cards = ttk.Frame(side)
        cards.pack(fill="x")
        for i in range(4):
            cards.columnconfigure(i, weight=1)
        c, self.v_status = self._card(cards, "STATUS")
        c.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        c, self.v_pose = self._card(cards, "DETECTED POSE")
        c.grid(row=0, column=1, sticky="ew", padx=6)
        c, self.v_fps = self._card(cards, "FPS")
        c.grid(row=0, column=2, sticky="ew", padx=6)
        c, self.v_count = self._card(cards, "GESTURES")
        c.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        self.v_status.configure(text="Stopped", foreground=MUTED)
        self.v_count.configure(text="0")

        ttk.Label(side, text="Controls", style="Head.TLabel").pack(anchor="w", pady=(16, 6))
        row = ttk.Frame(side)
        row.pack(fill="x")
        self.btn_start = ttk.Button(row, text="▶  Start Camera", style="Accent.TButton",
                                    command=self.toggle_camera)
        self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(row, text="⏸  Pause", command=self.toggle_pause,
                                    state="disabled")
        self.btn_pause.pack(side="left", padx=8)
        ttk.Button(row, text="📷  Screenshot", command=lambda: self.screenshot(0)).pack(side="left")
        ttk.Button(row, text="Screenshot in 3 s",
                   command=lambda: self.screenshot(3)).pack(side="left", padx=8)
        row2 = ttk.Frame(side)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Button(row2, text="Guided screenshot tour", command=self.start_tour).pack(side="left")
        ttk.Label(row2, text="walks through every gesture and saves a screenshot of each",
                  foreground=MUTED).pack(side="left", padx=10)

        opts = ttk.Frame(side)
        opts.pack(fill="x", pady=(10, 0))
        ttk.Label(opts, text="Camera").pack(side="left")
        ttk.Spinbox(opts, from_=0, to=9, width=3, textvariable=self.camera_var).pack(
            side="left", padx=(6, 16))
        ttk.Checkbutton(opts, text="Demo mode (log gestures, don't control PC)",
                        variable=self.demo_var, command=self.on_demo).pack(side="left")

        sens = ttk.Frame(side)
        sens.pack(fill="x", pady=(10, 0))
        ttk.Label(sens, text="Scroll sensitivity").pack(side="left")
        ttk.Label(sens, text="slow").pack(side="left", padx=(12, 4))
        # larger step = slower, so the scale runs from 0.20 (slow) to 0.03 (fast)
        ttk.Scale(sens, from_=0.20, to=0.03, variable=self.sens_var, length=220,
                  command=self.on_sensitivity).pack(side="left")
        ttk.Label(sens, text="fast").pack(side="left", padx=4)

        ttk.Label(side, text="Gesture guide", style="Head.TLabel").pack(anchor="w", pady=(18, 6))
        self.guide = ttk.Treeview(side, columns=("gesture", "action", "used"),
                                  show="headings", height=len(GESTURE_GUIDE), selectmode="none")
        for col, text, w in (("gesture", "Gesture", 210), ("action", "Action", 170),
                             ("used", "Times used", 90)):
            self.guide.heading(col, text=text, anchor="w")
            self.guide.column(col, width=w, anchor="w")
        self.guide_rows = list(GESTURE_GUIDE)
        if self.video_mode:                # demo_config.toml maps these differently
            demo = {"palm": "Next / previous page", "ok": "Detected only (demo)"}
            self.guide_rows = [(p_, g_, demo.get(p_, a_)) for p_, g_, a_ in GESTURE_GUIDE]
        for pose, gesture, action in self.guide_rows:
            self.guide.insert("", "end", iid=pose, values=(gesture, action, 0))
        self.guide.tag_configure("active", background="#2f4a3a")
        self.guide.pack(fill="x")

        ttk.Label(side, text="Session", style="Head.TLabel").pack(anchor="w", pady=(18, 6))
        self.session = ttk.Label(side, text="Not started", foreground=MUTED,
                                 justify="left", font=(FONT, 10))
        self.session.pack(anchor="w")

        tips = ("Tips:  keep your hand 40-80 cm from the camera in good light.  "
                "Lower your hand to 'lift' it.  Space = pause,  F12 = screenshot.")
        ttk.Label(side, text=tips, foreground=MUTED, wraplength=520,
                  justify="left").pack(anchor="w", side="bottom", pady=(10, 0))

    # ---- logging -------------------------------------------------------------
    def log(self, text: str, tag: str = "info") -> None:
        line = f"[{datetime.now():%H:%M:%S}] {text}"
        self.log_lines.append(line)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n", tag)
        if int(self.log_box.index("end-1c").split(".")[0]) > 500:
            self.log_box.delete("1.0", "2.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def export_log(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps({"log": self.log_lines, "gesture_counts": dict(self.stats)},
                                   indent=2))
        self.log(f"Log exported to {path.relative_to(HERE)}", "info")

    # ---- controls -----------------------------------------------------------
    def toggle_camera(self) -> None:
        if self.engine and self.engine.is_alive():
            self.engine.stop()
            self.btn_start.configure(text="Stopping…", state="disabled")
            return
        self.engine = GestureEngine(self.camera_var.get(), self.cfg, self.events,
                                    demo_mode=self.demo_var.get(),
                                    scroll_step=self.sens_var.get())
        self.engine.start()
        self.btn_start.configure(text="Starting…", state="disabled")
        self.v_status.configure(text="Starting", foreground=WARN)
        self.log(f"Opening camera {self.camera_var.get()}…")

    def toggle_pause(self) -> None:
        if self.engine and self.engine.is_alive():
            self.engine.toggle_pause()

    def on_demo(self) -> None:
        on = self.demo_var.get()
        if self.engine:
            self.engine.demo_mode = on
        self.log("Demo mode ON: gestures are only logged." if on
                 else "Demo mode OFF: gestures control the computer.", "warn")

    def on_sensitivity(self, _value=None) -> None:
        if self.engine:
            self.engine.set_scroll_step(self.sens_var.get())

    def screenshot(self, delay: int, name: str | None = None) -> None:
        """Save a PNG of this window (uses grim on Wayland, import on X11)."""
        if delay:
            self.log(f"Screenshot in {delay} s - show your gesture now!", "warn")
            self.after(delay * 1000, lambda: self.screenshot(0))
            return
        SHOTS_DIR.mkdir(exist_ok=True)
        # numbered + named after what is on screen, e.g. 03_pointing.png
        n = len(list(SHOTS_DIR.glob("*.png"))) + 1
        state = ("camera_off" if not (self.engine and self.engine.is_alive())
                 else "paused" if self.engine.paused
                 else POSE_LABELS.get(self.current_pose, "app").lower().replace(" ", "_"))
        path = SHOTS_DIR / f"{n:02d}_{name or state}.png"
        self.update_idletasks()
        cmd = None
        if shutil.which("grim") and shutil.which("hyprctl"):
            geo = self._hypr_geometry()
            cmd = ["grim"] + (["-g", geo] if geo else []) + [str(path)]
        elif shutil.which("import"):
            cmd = ["import", "-window", str(self.winfo_id()), str(path)]
        if not cmd:
            messagebox.showerror(APP_NAME, "No screenshot tool found (install grim).")
            return
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            self.log(f"Screenshot saved: {path.relative_to(HERE)}", "info")
        else:
            self.log(f"Screenshot failed: {res.stderr.strip()[:80]}", "error")

    def _hypr_client(self) -> dict | None:
        """This window's entry in `hyprctl clients` (matched by process id)."""
        if not shutil.which("hyprctl"):
            return None
        try:
            clients = json.loads(subprocess.run(["hyprctl", "clients", "-j"],
                                                capture_output=True, text=True).stdout)
        except (json.JSONDecodeError, OSError):
            return None
        return next((c for c in clients if c.get("pid") == os.getpid()), None)

    def _hypr_geometry(self) -> str | None:
        c = self._hypr_client()
        if not c:
            return None
        (x, y), (w, h) = c["at"], c["size"]
        return f"{x},{y} {w}x{h}"

    def _hypr_float(self) -> None:
        """On Hyprland, float + center the window at its designed size
        (a tiled half-screen slot is too narrow for the two-column layout)."""
        c = self._hypr_client()
        if not c or c.get("floating"):
            return
        win = f"address:{c['address']}"
        for lua in (f'hl.dsp.window.float({{ window = "{win}", action = "on" }})',
                    f'hl.dsp.window.resize({{ window = "{win}", x = 1240, y = 760 }})',
                    f'hl.dsp.window.center({{ window = "{win}" }})'):
            subprocess.run(["hyprctl", "dispatch", lua],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---- events from the engine thread ------------------------------------
    def poll_events(self) -> None:
        frame = None
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "frame":
                    frame = data                  # only draw the newest frame
                else:
                    self.handle_event(kind, data)
        except queue.Empty:
            pass
        if frame:
            self.photo = tk.PhotoImage(data=frame["ppm"], format="ppm")
            self.video.configure(image=self.photo, text="")
            if frame["pose"] != self.current_pose:
                self.pose_since = time.monotonic()
            self.current_pose = frame["pose"]
            self.v_pose.configure(text=POSE_LABELS.get(frame["pose"], frame["pose"]))
            self.v_fps.configure(text=f"{frame['fps']:.0f}")
            for pose, *_ in GESTURE_GUIDE:
                self.guide.item(pose, tags=("active",) if pose == frame["pose"] else ())
            if self.tour_i is not None:
                self.update_idletasks()          # make sure this frame is on screen
                self.tour_check()
        self.after(15, self.poll_events)

    def handle_event(self, kind: str, data: dict) -> None:
        if kind == "started":
            self.session_start = time.monotonic()
            self.btn_start.configure(text="■  Stop Camera", style="Danger.TButton",
                                     state="normal")
            self.btn_pause.configure(state="normal", text="⏸  Pause")
            self.v_status.configure(text="Running", foreground=ACCENT)
            self.log("Camera started. Show your hand to the camera.", "info")
            if self.tour_i is not None and self.tour_i == self.tour_first - 2:
                self.tour_next()
        elif kind == "stopped":
            self.btn_start.configure(text="▶  Start Camera", style="Accent.TButton",
                                     state="normal")
            self.btn_pause.configure(state="disabled")
            self.v_status.configure(text="Stopped", foreground=MUTED)
            self.v_pose.configure(text="-")
            self.v_fps.configure(text="-")
            self.photo = None
            self.video.configure(image=self.blank, text="Camera is off\n\nPress  ▶ Start Camera")
            self.banner.configure(text="")
            self.log("Camera stopped.", "info")
        elif kind == "paused":
            p = data["paused"]
            self.btn_pause.configure(text="▶  Resume" if p else "⏸  Pause")
            self.v_status.configure(text="Paused" if p else "Running",
                                    foreground=WARN if p else ACCENT)
            self.log("Gestures paused." if p else "Gestures resumed.", "warn")
            self.tour_events["paused" if p else "resumed"] = time.monotonic()
            self.stats["Pause/resume"] += 1
            self.v_count.configure(text=str(sum(self.stats.values())))
            self._update_guide_counts()
        elif kind == "action":
            gesture = data["gesture"]
            self.stats[gesture] += 1
            self.last_action, self.last_action_t = gesture, time.monotonic()
            self.tour_actions += 1
            self.tour_gestures[gesture] += 1
            self.v_count.configure(text=str(sum(self.stats.values())))
            self.banner.configure(text=f"  ➜  {gesture}", fg=BLUE if data["demo"] else ACCENT)
            self.after(1200, lambda t=gesture: self.banner.cget("text").endswith(t)
                       and self.banner.configure(text=""))
            self.log(f"{gesture:<13} -> {data['spec']}" + ("   [demo]" if data["demo"] else ""),
                     "demo" if data["demo"] else "action")
            self._update_guide_counts()
        elif kind == "error":
            self.log(data["text"], "error")
            messagebox.showerror(APP_NAME, data["text"])
            self.btn_start.configure(text="▶  Start Camera", style="Accent.TButton",
                                     state="normal")
            self.v_status.configure(text="Error", foreground=DANGER)

    # ---- guided screenshot tour --------------------------------------------
    def start_tour(self) -> None:
        """Walk through every gesture; a screenshot is saved when each is detected."""
        if self.video_mode:
            self.log(f"Demo video: live control, config {CONFIG_PATH.name}", "warn")
        elif not self.demo_var.get():
            self.demo_var.set(True)       # so swipes/OK sign don't leave this window
            self.on_demo()
        self.log("Guided tour started: follow the yellow instructions.", "warn")
        self.tour_label.pack(fill="x", pady=(6, 0), before=self.video)
        self.tour_label.configure(text="Starting the camera…")
        self.tour_i = self.tour_first - 2      # tour_next() moves to the first step
        if self.engine and self.engine.is_alive():
            self.tour_next()
        else:
            self.toggle_camera()           # tour_next() runs on the "started" event

    def tour_next(self) -> None:
        self.tour_i += 1
        if self.tour_i >= len(self.tour_steps):
            self.tour_i = None
            self.tour_label.configure(text="Demo complete - thank you!" if self.video_mode
                                      else "Tour complete - screenshots saved in screenshots/")
            self.log("Guided tour complete.", "warn")
            print("TOUR DONE", flush=True)
            if self.tour_auto:
                self.after(2500, self.on_close)
            return
        _, text = self.tour_steps[self.tour_i]
        self.tour_label.configure(text=f"STEP {self.tour_i + 1}/{len(self.tour_steps)}:  {text}")
        self.tour_step_t0 = time.monotonic()
        self.tour_actions = 0
        self.tour_gestures.clear()
        self.tour_events.clear()

    def tour_check(self) -> None:
        if self.tour_i is None or self.tour_i < self.tour_first - 1:
            return
        name, _ = self.tour_steps[self.tour_i]
        seen = self.tour_gestures
        now = time.monotonic()
        elapsed = now - self.tour_step_t0
        pose, held = self.current_pose, now - self.pose_since
        recent = now - self.last_action_t < 0.6
        done = {
            "camera_running": pose == g.NONE and elapsed > 3,
            "pointing": pose == g.POINT and recent and self.last_action.startswith("Scroll")
                        and self.tour_actions >= 2,
            "open_palm": recent and self.last_action.startswith("Swipe"),
            "two_fingers": pose == g.TWO and recent and self.last_action.startswith("Volume")
                           and self.tour_actions >= 2,
            "ok_sign": pose == g.OK and held > 0.45,           # progress bar half full
            "fist_paused": "paused" in self.tour_events
                           and now - self.tour_events["paused"] > 0.3,
            "resumed": "resumed" in self.tour_events
                       and now - self.tour_events["resumed"] > 1.0,
            "session_summary": elapsed > 2.5,
            # video steps
            "intro": pose != g.NONE and elapsed > 3,
            "scroll_down": seen["Scroll down"] >= 12,
            "scroll_up": seen["Scroll up"] >= 8,
            "swipe": seen["Swipe left"] + seen["Swipe right"] >= 1 and elapsed > 3,
            "volume": seen["Volume up"] + seen["Volume down"] >= 4,
            "finish": elapsed > 4,
        }[name]
        if elapsed < (2.5 if self.video_mode else 1.5) and name != "ok_sign":
            done = False                   # give time to read the instruction
        if not done and elapsed > STEP_TIMEOUT:
            self.log(f"Step '{name}' not detected in time; taking the screenshot anyway.", "warn")
            done = True
        if done:
            if not self.video_mode:
                self.screenshot(0, name)
            self.tour_next()

    def _update_guide_counts(self) -> None:
        groups = {"point": ("Scroll",), "palm": ("Swipe",), "two": ("Volume",),
                  "ok": ("OK sign",), "fist": ("Pause",)}
        for pose, gesture, action in self.guide_rows:
            n = sum(v for k, v in self.stats.items() if k.startswith(groups[pose])) \
                if groups[pose] else 0
            self.guide.item(pose, values=(gesture, action, n))

    def tick_clock(self) -> None:
        self.clock.configure(text=datetime.now().strftime("%a %d %b  %H:%M:%S"))
        if self.session_start and self.engine and self.engine.is_alive():
            secs = int(time.monotonic() - self.session_start)
            top = ", ".join(f"{k} ×{v}" for k, v in self.stats.most_common(3)) or "none yet"
            self.session.configure(
                text=f"Running for {secs // 60:02d}:{secs % 60:02d}   •   "
                     f"{sum(self.stats.values())} gestures\nMost used: {top}")
        self.after(1000, self.tick_clock)

    def on_close(self) -> None:
        if self.engine and self.engine.is_alive():
            self.engine.stop()
            self.engine.join(timeout=2)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
