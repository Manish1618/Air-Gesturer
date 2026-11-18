"""Hand pose classification and gesture state machines.

All motion is measured in "palm units" (wrist to middle-finger knuckle) so
gestures feel the same whether the hand is near or far from the camera.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

POINT, TWO, PALM, FIST, OK, OTHER, NONE = "point", "two", "palm", "fist", "ok", "other", "none"

# (tip, pip) landmark pairs for index, middle, ring, little fingers.
_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(lm) -> float:
    return max(_dist(lm[0], lm[9]), 1e-3)


def classify(lm) -> str:
    """Return the pose of one hand from its 21 landmarks.

    A finger counts as extended when its tip is clearly farther from the wrist
    than its middle joint, which works for tilted hands, unlike a plain y test.
    """
    wrist = lm[0]
    up = [_dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.15 for tip, pip in _FINGERS]
    # OK sign: thumb and index tips touch, the other three fingers stay up.
    if all(up[1:]) and _dist(lm[4], lm[8]) < palm_size(lm) * 0.35:
        return OK
    if up == [True, False, False, False]:
        return POINT
    if up == [True, True, False, False]:
        return TWO
    if all(up):
        return PALM
    if not any(up):
        return FIST
    return OTHER


class PoseFilter:
    """Debounce poses: a new pose must persist for `frames` frames."""

    def __init__(self, frames: int = 3):
        self.frames = frames
        self.stable = NONE
        self._candidate = NONE
        self._count = 0

    def update(self, pose: str) -> str:
        if pose == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = pose, 1
        if self._count >= self.frames:
            self.stable = pose
        return self.stable


class Smoother:
    """Exponential moving average for a 2D point."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: tuple[float, float] | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self.value is None:
            self.value = (x, y)
        else:
            a = self.alpha
            self.value = (a * x + (1 - a) * self.value[0], a * y + (1 - a) * self.value[1])
        return self.value


@dataclass
class Scroller:
    """Turns pointing-finger motion into wheel ticks (vertical and horizontal)."""

    step: float = 0.08            # palm units of travel per wheel tick
    deadzone: float = 0.006       # per-frame motion ignored as jitter
    horizontal: bool = True
    max_ticks: int = 6
    _last: tuple[float, float] | None = None
    _carry: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _axis: int | None = None      # 0 = x, 1 = y; locked per gesture

    def reset(self) -> None:
        self._last, self._carry, self._axis = None, [0.0, 0.0], None

    def update(self, x: float, y: float, palm: float) -> tuple[int, int]:
        """Return (horizontal_ticks, vertical_ticks); positive = right / down."""
        if self._last is None:
            self._last = (x, y)
            return 0, 0
        d = [(x - self._last[0]) / palm, (y - self._last[1]) / palm]
        self._last = (x, y)
        if self._axis is None:
            moved = [abs(v) for v in d]
            if max(moved) < self.deadzone * 2:
                return 0, 0
            self._axis = 0 if self.horizontal and moved[0] > moved[1] * 1.5 else 1
        ax = self._axis
        if abs(d[ax]) < self.deadzone:
            return 0, 0
        self._carry[ax] += d[ax]
        ticks = int(self._carry[ax] / self.step)
        if not ticks:
            return 0, 0
        self._carry[ax] -= ticks * self.step
        ticks = max(-self.max_ticks, min(self.max_ticks, ticks))
        return (ticks, 0) if ax == 0 else (0, ticks)


@dataclass
class SwipeDetector:
    """Detects a quick open-palm swipe; returns 'left'/'right'/'up'/'down'."""

    distance: float = 1.3   # palm units the hand must travel
    window: float = 0.35    # seconds the travel must happen within
    cooldown: float = 0.8
    _hist: deque = field(default_factory=deque)
    _until: float = 0.0

    def reset(self) -> None:
        self._hist.clear()

    def update(self, x: float, y: float, palm: float, now: float) -> str | None:
        if now < self._until:
            return None
        self._hist.append((now, x, y))
        while self._hist and now - self._hist[0][0] > self.window:
            self._hist.popleft()
        _, x0, y0 = self._hist[0]
        dx, dy = (x - x0) / palm, (y - y0) / palm
        if max(abs(dx), abs(dy)) < self.distance:
            return None
        if abs(dx) > abs(dy) * 1.8:
            direction = "right" if dx > 0 else "left"
        elif abs(dy) > abs(dx) * 1.8:
            direction = "down" if dy > 0 else "up"
        else:
            return None
        self._hist.clear()
        self._until = now + self.cooldown
        return direction


@dataclass
class Stepper:
    """Fires one step per `step` palm units of vertical travel (e.g. volume)."""

    step: float = 0.35
    _last_y: float | None = None
    _carry: float = 0.0

    def reset(self) -> None:
        self._last_y, self._carry = None, 0.0

    def update(self, y: float, palm: float) -> int:
        """Return steps moved; positive = down."""
        if self._last_y is None:
            self._last_y = y
            return 0
        self._carry += (y - self._last_y) / palm
        self._last_y = y
        steps = int(self._carry / self.step)
        self._carry -= steps * self.step
        return steps


@dataclass
class HoldDetector:
    """Fires once when a pose is held for `hold` seconds, then needs a release."""

    hold: float = 1.0
    _since: float | None = None
    _fired: bool = False

    def update(self, active: bool, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not active:
            self._since, self._fired = None, False
            return False
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.hold:
            self._fired = True
            return True
        return False

    def progress(self, now: float) -> float:
        if self._since is None or self._fired:
            return 0.0
        return min(1.0, (now - self._since) / self.hold)
