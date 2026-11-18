# Air Gesture Controller: Brief Explanation

**Air Gesture Controller** is a Python desktop application that lets you control your
computer by moving your hand in the air in front of a normal webcam, without touching
the mouse or keyboard.

**How it works:** The app reads live video from the webcam with **OpenCV** and passes
each frame to Google's **MediaPipe Hands** model, which finds 21 points on the hand
(wrist, knuckles, fingertips). From these points the program works out which fingers
are raised and recognises five poses: pointing, open palm, two fingers, OK sign and
fist. It then tracks how the hand moves and turns that movement into actions:

| Gesture | Action |
| --- | --- |
| Point with index finger and move up/down | Scroll the page |
| Quick open-palm swipe left/right | Switch workspace |
| Two fingers up, move up/down | Volume up / down |
| Hold OK sign | Close the active window |
| Hold a fist for 4 seconds | Pause / resume gestures |

All movement is measured relative to the size of the palm, so the gestures work whether
the hand is near or far from the camera.

**Interface:** A **Tkinter** GUI shows the live camera with the hand skeleton drawn on
it, the detected gesture, the frame rate (about 30 FPS), Start/Stop/Pause buttons, a
scroll-sensitivity slider, a gesture guide with usage counts, a colour-coded activity
log (exportable as JSON) and a built-in screenshot button. The camera processing runs
in a separate **background thread** that sends results to the GUI through a
thread-safe queue, so the window never freezes. A **Demo mode** logs gestures without
controlling the PC.

**Real-world use:** It is useful for hands-free scrolling while cooking or reading,
for controlling presentations from a distance, and for users who find a mouse hard to
use. All processing happens locally, and no video is saved or sent anywhere.

**Concepts used:** computer vision, machine-learning inference, geometry-based
classification, state machines, object-oriented programming, multithreading, GUI
programming, file handling (TOML config, JSON logs) and error handling.

**Run:** `.venv/bin/python air_gesture_app.py`
