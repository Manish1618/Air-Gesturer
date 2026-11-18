# Air Gesturer

Control your computer with hand gestures in the air, using an ordinary webcam.
Python application development project.

The project has two parts:

| Part | Where | What it does |
| --- | --- | --- |
| **Desktop app** (`air_gesture_app.py`) | Your Linux PC | Python + OpenCV + MediaPipe + Tkinter. Controls the whole computer: scrolls any window, switches workspaces, changes the system volume, closes windows. |
| **Web version** (`web/`) | Any browser, hosted on Render | The same gesture algorithm (ported to JavaScript) controls a playground on the page, using the visitor's own webcam. Nothing is uploaded. |

## Gestures

| Gesture | Desktop app | Web version |
| --- | --- | --- |
| ☝️ Point with the index finger and move up/down | Scroll / arrow keys | Scroll the playground |
| 🖐️ Quick open-palm swipe left/right | Next / previous workspace | Next / previous page |
| ✌️ Index + middle finger up, move up/down | System volume up/down | On-page volume |
| 👌 OK sign, hold 0.8 s | Close the active window | Dismiss the notification |
| ✊ Hold a fist | Pause / resume (4 s) | Pause / resume (2 s) |

Motion is measured relative to the palm size, so gestures work near or far from
the camera. Lower your hand or change pose to "lift" it, like lifting a finger
off a touchpad.

## How it works

1. **Capture** a webcam frame and mirror it.
2. **Detect** 21 hand landmarks with Google MediaPipe Hands.
3. **Measure** movement in *palm units* (wrist to middle-finger knuckle).
4. **Classify** the pose from which fingers are extended (tip farther than 1.15× the middle joint from the wrist).
5. **Debounce**: a new pose must last 2 frames.
6. **Track** motion with state machines (`Scroller`, `SwipeDetector`, `Stepper`, `HoldDetector`).
7. **Act**: run the action mapped in `config.toml` (desktop) or update the page (web).

`gestures.py` and `web/gestures.js` contain the same algorithm. The JavaScript
port was checked against the Python module on 3,000 random hand poses and
motion sequences with identical results.

## Desktop app

Requires Linux (developed on Arch / Omarchy with Hyprland), Python 3.12 and a webcam.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python air_gesture_app.py
```

Or `./start.sh` (uses `uv`) for the original command-line version `air_scroll.py`
with an OpenCV preview window (`--no-preview`, `--dry-run`, `--camera 1`).

The GUI shows the live camera with the hand skeleton, status / pose / FPS / gesture
cards, Start / Stop / Pause, demo mode (log only), a scroll-sensitivity slider, a
gesture guide with usage counts, a colour-coded activity log (export to JSON) and a
screenshot button. Camera processing runs in a background thread.

### System setup for desktop actions

Scrolling and key presses use `ydotool`, which writes to `/dev/uinput`:

```sh
sudo pacman -S ydotool
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
  | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo usermod -aG input "$USER"
sudo udevadm control --reload && sudo udevadm trigger /dev/uinput
# log out and back in so the new group applies, then:
systemctl --user enable --now ydotool
```

Workspace and window actions use `hyprctl`; volume uses `wpctl`.

### Configuration

Edit `config.toml` to tune sensitivity or remap actions. Actions are strings:

- `hypr:hl.dsp.focus({ workspace = "e+1" })` runs `hyprctl dispatch ...`
- `key:alt+Left` presses a key combo
- `cmd:playerctl play-pause` runs a shell command
- `""` does nothing

## Web version

Static site in `web/`: `index.html`, `style.css`, `app.js`, `gestures.js` (the gesture logic) and `hands.js` (hand-skeleton drawings).
Hand tracking uses MediaPipe Tasks Vision (WebAssembly + WebGL) loaded from a CDN.
Video stays in the browser tab. Needs a browser with WebGL and camera access
(Chrome, Edge, Firefox; HTTPS or localhost).

Run locally:

```sh
python3 -m http.server 8000 --directory web
# open http://localhost:8000
```

### Deploy on Render

`render.yaml` describes a free **Static Site** that publishes the `web/` folder.
On [render.com](https://render.com): **New → Blueprint**, connect this repository,
and apply. Or **New → Static Site** with *Publish directory* `web` and the
build command `echo ok` (Render needs one, even though nothing is built).

## Files

| File | Purpose |
| --- | --- |
| `air_gesture_app.py` | Desktop GUI app (Tkinter) + gesture engine thread |
| `gestures.py` | Pose classification and gesture state machines |
| `actions.py` | Desktop actions: ydotool, hyprctl, wtype, shell commands |
| `air_scroll.py` | Original command-line version with OpenCV preview |
| `config.toml` | Sensitivity and gesture → action mapping |
| `web/` | Browser version (deployed to Render) |
| `Air_Gesture_Controller_Documentation.pdf` | Project documentation |
| `EXPLANATION.md` | Short explanation of the application |
| `screenshots/`, `demo_video.mp4` | Output screenshots and demo video |
