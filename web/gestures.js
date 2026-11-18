// Hand pose classification and gesture state machines.
// A direct JavaScript port of gestures.py from the Air Gesture Controller
// desktop app, so the browser version behaves the same way.
//
// All motion is measured in "palm units" (wrist to middle-finger knuckle) so
// gestures feel the same whether the hand is near or far from the camera.

export const POINT = "point", TWO = "two", PALM = "palm", FIST = "fist",
  OK = "ok", OTHER = "other", NONE = "none";

// (tip, pip) landmark pairs for index, middle, ring, little fingers.
const FINGERS = [[8, 6], [12, 10], [16, 14], [20, 18]];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export const palmSize = (lm) => Math.max(dist(lm[0], lm[9]), 1e-3);

/** Return the pose of one hand from its 21 landmarks.
 *  A finger counts as extended when its tip is clearly farther from the wrist
 *  than its middle joint, which also works for tilted hands. */
export function classify(lm) {
  const wrist = lm[0];
  const up = FINGERS.map(([tip, pip]) => dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.15);
  // OK sign: thumb and index tips touch, the other three fingers stay up.
  if (up[1] && up[2] && up[3] && dist(lm[4], lm[8]) < palmSize(lm) * 0.35) return OK;
  const key = up.map(Number).join("");
  if (key === "1000") return POINT;
  if (key === "1100") return TWO;
  if (key === "1111") return PALM;
  if (key === "0000") return FIST;
  return OTHER;
}

/** Debounce poses: a new pose must persist for `frames` frames. */
export class PoseFilter {
  constructor(frames = 2) { this.frames = frames; this.stable = NONE; this.candidate = NONE; this.count = 0; }
  update(pose) {
    if (pose === this.candidate) this.count++;
    else { this.candidate = pose; this.count = 1; }
    if (this.count >= this.frames) this.stable = pose;
    return this.stable;
  }
}

/** Exponential moving average for a 2D point. */
export class Smoother {
  constructor(alpha = 0.5) { this.alpha = alpha; this.value = null; }
  reset() { this.value = null; }
  update(x, y) {
    if (!this.value) this.value = [x, y];
    else {
      const a = this.alpha;
      this.value = [a * x + (1 - a) * this.value[0], a * y + (1 - a) * this.value[1]];
    }
    return this.value;
  }
}

/** Turns pointing-finger motion into scroll ticks (vertical and horizontal). */
export class Scroller {
  constructor({ step = 0.08, deadzone = 0.006, horizontal = true, maxTicks = 6 } = {}) {
    Object.assign(this, { step, deadzone, horizontal, maxTicks });
    this.reset();
  }
  reset() { this.last = null; this.carry = [0, 0]; this.axis = null; }
  /** Returns [horizontalTicks, verticalTicks]; positive = right / down. */
  update(x, y, palm) {
    if (!this.last) { this.last = [x, y]; return [0, 0]; }
    const d = [(x - this.last[0]) / palm, (y - this.last[1]) / palm];
    this.last = [x, y];
    if (this.axis === null) {
      const moved = d.map(Math.abs);
      if (Math.max(...moved) < this.deadzone * 2) return [0, 0];
      this.axis = this.horizontal && moved[0] > moved[1] * 1.5 ? 0 : 1;
    }
    const ax = this.axis;
    if (Math.abs(d[ax]) < this.deadzone) return [0, 0];
    this.carry[ax] += d[ax];
    let ticks = Math.trunc(this.carry[ax] / this.step);
    if (!ticks) return [0, 0];
    this.carry[ax] -= ticks * this.step;
    ticks = Math.max(-this.maxTicks, Math.min(this.maxTicks, ticks));
    return ax === 0 ? [ticks, 0] : [0, ticks];
  }
}

/** Detects a quick open-palm swipe; returns "left" / "right" / "up" / "down". */
export class SwipeDetector {
  constructor({ distance = 0.55, window = 0.5, cooldown = 0.8 } = {}) {
    Object.assign(this, { distance, window, cooldown });
    this.hist = []; this.until = 0;
  }
  reset() { this.hist = []; }
  update(x, y, palm, now) {
    if (now < this.until) return null;
    this.hist.push([now, x, y]);
    while (this.hist.length && now - this.hist[0][0] > this.window) this.hist.shift();
    const [, x0, y0] = this.hist[0];
    const dx = (x - x0) / palm, dy = (y - y0) / palm;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < this.distance) return null;
    let direction;
    if (Math.abs(dx) > Math.abs(dy) * 1.8) direction = dx > 0 ? "right" : "left";
    else if (Math.abs(dy) > Math.abs(dx) * 1.8) direction = dy > 0 ? "down" : "up";
    else return null;
    this.hist = [];
    this.until = now + this.cooldown;
    return direction;
  }
}

/** Fires one step per `step` palm units of vertical travel (e.g. volume). */
export class Stepper {
  constructor(step = 0.15) { this.step = step; this.reset(); }
  reset() { this.lastY = null; this.carry = 0; }
  /** Returns steps moved; positive = down. */
  update(y, palm) {
    if (this.lastY === null) { this.lastY = y; return 0; }
    this.carry += (y - this.lastY) / palm;
    this.lastY = y;
    const steps = Math.trunc(this.carry / this.step);
    this.carry -= steps * this.step;
    return steps;
  }
}

/** Fires once when a pose is held for `hold` seconds, then needs a release. */
export class HoldDetector {
  constructor(hold = 1.0) { this.hold = hold; this.since = null; this.fired = false; }
  update(active, now) {
    if (!active) { this.since = null; this.fired = false; return false; }
    if (this.since === null) this.since = now;
    if (!this.fired && now - this.since >= this.hold) { this.fired = true; return true; }
    return false;
  }
  progress(now) {
    if (this.since === null || this.fired) return 0;
    return Math.min(1, (now - this.since) / this.hold);
  }
}
