// Small hand-skeleton drawings (same 21-point layout as MediaPipe) used for
// the gesture legend and the idle animation. Coordinates are in a 100x100 box.

export const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

const WRIST = [50, 94];
const MCP = { index: [37, 52], middle: [49, 49], ring: [60, 51], pinky: [70, 57] };
const DIR = { index: [-0.13, -1], middle: [0, -1], ring: [0.12, -1], pinky: [0.25, -1] };
const LEN = { index: [13, 9, 8], middle: [14, 10, 8], ring: [13, 9, 7], pinky: [10, 7, 6] };

function finger(name, extended) {
  const [mx, my] = MCP[name], [dx, dy] = DIR[name], [a, b, c] = LEN[name];
  if (extended) {
    return [[mx + dx * a, my + dy * a], [mx + dx * (a + b), my + dy * (a + b)],
      [mx + dx * (a + b + c), my + dy * (a + b + c)]];
  }
  // folded toward the palm: seen from the front it's just a short knuckle
  const k = [mx + dx * a * 0.45, my + dy * a * 0.45];
  return [k, k, k];
}

const THUMB_OUT = [[37, 82], [28, 72], [21, 63], [16, 55]];
const THUMB_IN = [[38, 82], [35, 72], [41, 66], [47, 64]];

/** 21 landmarks for a named pose. */
export function poseLandmarks(pose) {
  const up = {
    point: [1, 0, 0, 0], two: [1, 1, 0, 0], palm: [1, 1, 1, 1], fist: [0, 0, 0, 0], ok: [0, 1, 1, 1],
  }[pose];
  let thumb = pose === "palm" ? THUMB_OUT : THUMB_IN;
  const fingers = ["index", "middle", "ring", "pinky"].map((f, i) => finger(f, up[i]));
  if (pose === "ok") {                  // index curls round to meet the thumb tip
    fingers[0] = [[31, 42], [25, 44], [27, 51]];
    thumb = [[37, 82], [29, 73], [25, 63], [27, 54]];
  }
  const pts = [WRIST, ...thumb];
  ["index", "middle", "ring", "pinky"].forEach((f, i) => pts.push(MCP[f], ...fingers[i]));
  return pts;
}

/** Inline SVG of a pose. Folded finger joints are not drawn. */
export function poseSVG(pose, { bone = "currentColor", joint = "#FFC531", size = 48, stroke = 4, r = 3 } = {}) {
  const p = poseLandmarks(pose);
  const seen = new Set();
  const key = (i) => `${p[i][0].toFixed(1)},${p[i][1].toFixed(1)}`;
  const lines = BONES.filter(([a, b]) => key(a) !== key(b))
    .map(([a, b]) => `M${p[a][0]} ${p[a][1]}L${p[b][0]} ${p[b][1]}`).join("");
  const dots = p.map((_, i) => i).filter((i) => !seen.has(key(i)) && seen.add(key(i)))
    .map((i) => `<circle cx="${p[i][0]}" cy="${p[i][1]}" r="${r}"/>`).join("");
  return `<svg viewBox="6 22 80 78" width="${size}" height="${size}" aria-hidden="true">
    <path d="${lines}" fill="none" stroke="${bone}" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round"/>
    <g fill="${joint}">${dots}</g></svg>`;
}
