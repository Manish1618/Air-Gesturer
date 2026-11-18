"""Output backends: wheel scrolling, Hyprland dispatches, key combos, commands.

Action strings in config.toml use a prefix:
  hypr:<dispatch>            -> hyprctl dispatch <dispatch>; with a Lua config
                                (Hyprland 0.5x+) e.g. hypr:hl.dsp.focus({ workspace = "e+1" })
  key:<mod+mod+Key>          -> key combo via ydotool (wtype for unknown keys),
                                e.g. key:alt+Left
  cmd:<shell command>        -> run through /bin/sh
  (empty string)             -> do nothing
"""

from __future__ import annotations

import shlex
import shutil
import subprocess

# wtype modifier names; "super" is called "logo" there.
_MODS = {"shift": "shift", "ctrl": "ctrl", "control": "ctrl", "alt": "alt",
         "super": "logo", "logo": "logo", "win": "logo", "altgr": "altgr"}


def _spawn(argv: list[str]) -> None:
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


# Linux input keycodes for ydotool. Some apps (e.g. Evince) ignore wtype's
# virtual-keyboard events, but accept ydotool's, which look like a real keyboard.
_KEYCODES = {
    "shift": 42, "ctrl": 29, "control": 29, "alt": 56, "super": 125, "logo": 125, "win": 125,
    "altgr": 100, "escape": 1, "esc": 1, "tab": 15, "return": 28, "enter": 28, "space": 57,
    "backspace": 14, "delete": 111, "insert": 110, "home": 102, "end": 107,
    "up": 103, "down": 108, "left": 105, "right": 106,
    "page_up": 104, "prior": 104, "page_down": 109, "next": 109, "minus": 12, "equal": 13,
    **{f"f{i}": c for i, c in zip(range(1, 11), range(59, 69))}, "f11": 87, "f12": 88,
    **{str(d): c for d, c in zip("1234567890", range(2, 12))},
    **{ch: c for ch, c in zip("qwertyuiop", range(16, 26))},
    **{ch: c for ch, c in zip("asdfghjkl", range(30, 39))},
    **{ch: c for ch, c in zip("zxcvbnm", range(44, 51))},
}


def key_argv(combo: str) -> list[str]:
    *mods, key = [p.strip() for p in combo.split("+")]
    names = [n.lower() for n in mods + [key]]
    if shutil.which("ydotool") and all(n in _KEYCODES for n in names):
        codes = [_KEYCODES[n] for n in names]
        return (["ydotool", "key"] + [f"{c}:1" for c in codes]
                + [f"{c}:0" for c in reversed(codes)])
    argv = ["wtype"]
    for m in mods:
        argv += ["-M", _MODS[m.lower()]]
    argv += ["-k", key]
    for m in reversed(mods):
        argv += ["-m", _MODS[m.lower()]]
    return argv


class Actions:
    def __init__(self, invert_scroll: bool = False, dry_run: bool = False):
        self.invert = invert_scroll
        self.dry_run = dry_run
        self.can_scroll = bool(shutil.which("ydotool"))
        self.scroll_error = "" if self.can_scroll else "ydotool not installed"

    def scroll(self, right: int, down: int) -> bool:
        """Send wheel ticks. Positive = scroll right / down."""
        if not (right or down):
            return True
        if self.invert:
            right, down = -right, -down
        if self.dry_run:
            print(f"scroll right={right} down={down}")
            return True
        if not self.can_scroll:
            return False
        # ydotool: REL_HWHEEL positive = right, REL_WHEEL negative = down.
        try:
            subprocess.run(["ydotool", "mousemove", "--wheel", "--", str(right), str(-down)],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1)
        except subprocess.CalledProcessError as e:
            lines = (e.stderr or b"").decode().strip().splitlines()
            self.scroll_error = f"ydotool: {lines[-1] if lines else 'failed'} (is ydotoold running?)"
            return False
        except subprocess.TimeoutExpired:
            self.scroll_error = "ydotool timed out (is ydotoold running?)"
            return False
        self.scroll_error = ""
        return True

    def run(self, spec: str) -> None:
        spec = (spec or "").strip()
        if not spec:
            return
        kind, _, arg = spec.partition(":")
        if kind == "hypr":
            argv = ["hyprctl", "dispatch", arg.strip()]
        elif kind == "key":
            argv = key_argv(arg)
        elif kind == "cmd":
            argv = ["/bin/sh", "-c", arg]
        else:
            print(f"Unknown action {spec!r}; use hypr:, key: or cmd:")
            return
        if self.dry_run:
            print("action:", shlex.join(argv))
            return
        _spawn(argv)

    def notify(self, text: str) -> None:
        if not self.dry_run and shutil.which("notify-send"):
            _spawn(["notify-send", "-a", "Air Scroll", "-t", "1500", "Air Scroll", text])
