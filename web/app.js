// Air Gesturer (web): webcam hand gestures that control the page.
// Hand landmarks come from MediaPipe Hand Landmarker running locally in the
// browser (WebAssembly); gesture logic is the port of gestures.py.

import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import * as g from "./gestures.js";
import { BONES, poseSVG } from "./hands.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// same tuning as config.toml in the desktop app (fist hold shortened for the web)
const CFG = {
  confirmFrames: 2,
  scroll: { step: 0.08, deadzone: 0.006, horizontal: false, maxTicks: 6 },
  swipe: { distance: 0.55, window: 0.5, cooldown: 0.8 },
  twoStep: 0.15, okHold: 0.8, pauseHold: 2.0,
  scrollPixels: 38,                     // page pixels per scroll tick
};

const POSES = {
  [g.NONE]: ["No hand", "Raise one hand, palm facing the camera."],
  [g.POINT]: ["Pointing", "Move your finger up or down to scroll the sheet."],
  [g.PALM]: ["Open palm", "Swipe quickly left or right to turn the page."],
  [g.TWO]: ["Two fingers", "Raise or lower your hand to change the volume."],
  [g.FIST]: ["Fist", "Keep holding to pause."],
  [g.OK]: ["OK sign", "Keep holding to dismiss the message."],
  [g.OTHER]: ["Unknown pose", "Try one of the five gestures below."],
};

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");

// ---------------------------------------------------------------- state
let landmarker = null, stream = null, running = false, paused = false;
let lastVideoTime = -1, fps = 0, lastFrameT = 0, rafId = 0;
let pose = g.NONE, prevPose = null;
const used = { point: 0, palm: 0, two: 0, ok: 0, fist: 0 };
let volume = 50, page = 0;

const poseFilter = new g.PoseFilter(CFG.confirmFrames);
const smoother = new g.Smoother(0.5);
const scroller = new g.Scroller(CFG.scroll);
const swiper = new g.SwipeDetector(CFG.swipe);
const stepper = new g.Stepper(CFG.twoStep);
const fistHold = new g.HoldDetector(CFG.pauseHold);
const okHold = new g.HoldDetector(CFG.okHold);

// ---------------------------------------------------------------- static drawings
$("ghost").innerHTML = poseSVG("point", { size: 210, stroke: 2.6, r: 3 });
document.querySelectorAll(".legend li").forEach((li) => {
  li.querySelector("i").innerHTML = poseSVG(li.dataset.pose, { size: 40 });
});

// ---------------------------------------------------------------- UI helpers
let flashTimer;
function flash(text) {
  const f = $("flash");
  f.textContent = text; f.classList.add("show");
  clearTimeout(flashTimer); flashTimer = setTimeout(() => f.classList.remove("show"), 900);
}

function showStatus() {
  const [name, hint] = running ? (paused ? ["Paused", "Hold a fist again to resume."] : POSES[pose])
    : ["Camera off", "Start the camera, then raise one hand 40–80 cm away."];
  $("pose").textContent = name;
  $("pose").classList.toggle("paused", paused && running);
  $("hint").textContent = hint;
  document.querySelectorAll(".legend li").forEach((li) =>
    li.classList.toggle("live", running && !paused && li.dataset.pose === pose));
}

function count(key, label) {
  used[key]++;
  const li = document.querySelector(`.legend li[data-pose="${key}"]`);
  li.classList.add("tried");
  flash(label);
}

// ---------------------------------------------------------------- the sheet
const pages = [...document.querySelectorAll(".page")];
pages.forEach((p, i) => {
  const b = document.createElement("button");
  b.textContent = p.dataset.title;
  b.setAttribute("role", "tab");
  b.onclick = () => showPage(i);
  $("tabs").appendChild(b);
});
function showPage(i) {
  page = Math.max(0, Math.min(pages.length - 1, i));
  pages.forEach((p, j) => {
    p.classList.toggle("before", j < page);
    p.classList.toggle("after", j > page);
  });
  [...$("tabs").children].forEach((b, j) => b.setAttribute("aria-selected", j === page));
}
showPage(0);

function doScroll(ticks) {
  pages[page].scrollTop += ticks * CFG.scrollPixels;
  count("point", ticks > 0 ? "Scroll down" : "Scroll up");
}

function doSwipe(dir) {
  if (dir !== "left" && dir !== "right") return;
  const target = page + (dir === "left" ? 1 : -1);
  if (target < 0 || target >= pages.length) { flash(target < 0 ? "Already on the first page" : "That's the last page"); return; }
  showPage(target);
  count("palm", dir === "left" ? "Next page" : "Previous page");
}

function setVolume(v) {
  volume = Math.max(0, Math.min(100, v));
  $("vol-fill").style.width = volume + "%";
  $("vol-num").textContent = volume;
}
function doVolume(steps) {
  setVolume(volume - steps * 5);          // positive steps = hand moved down
  count("two", `Volume ${volume}`);
}

const MESSAGES = [
  ["One new message", "Make an OK sign and hold it to dismiss this."],
  ["Tip", "Gestures work best with your hand 40–80 cm from the camera, in good light."],
  ["Did you know?", "The tracker follows 21 points on your hand, about 30 times a second."],
];
let msgIndex = 0;
function doOk() {
  const n = $("notice");
  if (n.classList.contains("gone")) { flash("Nothing to dismiss"); return; }
  n.classList.add("gone");
  count("ok", "Message dismissed");
  setTimeout(() => {                       // a new message arrives a few seconds later
    msgIndex = (msgIndex + 1) % MESSAGES.length;
    $("notice-title").textContent = MESSAGES[msgIndex][0];
    $("notice-text").textContent = MESSAGES[msgIndex][1];
    n.classList.remove("gone");
  }, 4000);
}

function setPaused(p, byFist = false) {
  paused = p;
  $("pause").textContent = p ? "Resume" : "Pause";
  if (byFist) count("fist", p ? "Paused" : "Resumed");
  showStatus();
}

// ---------------------------------------------------------------- camera + model
async function loadModel() {
  if (landmarker) return landmarker;
  const files = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO", numHands: 1,
    minHandDetectionConfidence: 0.65, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6,
  });
  try { landmarker = await HandLandmarker.createFromOptions(files, opts("GPU")); }
  catch { landmarker = await HandLandmarker.createFromOptions(files, opts("CPU")); }
  return landmarker;
}

function idleMessage(text, isError = false) {
  const p = $("idle-note");
  p.textContent = text;
  p.classList.toggle("err", isError);
}

async function start() {
  const btn = $("start");
  btn.disabled = true; btn.textContent = "Starting…";
  idleMessage("Loading the hand tracker. This takes a few seconds the first time.");
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("This browser can't use a camera here. Open the page over HTTPS in Chrome, Edge or Firefox.");
    const c = document.createElement("canvas");
    if (!c.getContext("webgl2") && !c.getContext("webgl"))
      throw new Error("WebGL is turned off in this browser. Turn on hardware acceleration, or use Chrome, Edge or Firefox.");
    const modelP = loadModel();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    await modelP;
  } catch (err) {
    const msg = err.name === "NotAllowedError" ? "Camera access was blocked. Allow it from the icon in the address bar, then press Start camera again."
      : err.name === "NotFoundError" ? "No camera was found. Connect one and press Start camera again."
      : err.name === "NotReadableError" ? "Another app is using the camera. Close it and press Start camera again."
      : err.message || String(err);
    idleMessage(msg, true);
    btn.disabled = false; btn.textContent = "Start camera";
    stopStream();
    return;
  }
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  $("idle").hidden = true;
  $("stop").disabled = $("pause").disabled = false;
  running = true; paused = false; prevPose = null;
  showStatus();
  lastFrameT = performance.now();
  rafId = requestAnimationFrame(loop);
}

function stopStream() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null; video.srcObject = null;
}

function stop() {
  running = false; cancelAnimationFrame(rafId); stopStream();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("idle").hidden = false;
  const btn = $("start"); btn.disabled = false; btn.textContent = "Start camera";
  idleMessage("Camera stopped. Start it again whenever you like.");
  $("stop").disabled = $("pause").disabled = true;
  $("pause").textContent = "Pause";
  $("hold").classList.remove("show");
  $("fps").textContent = "";
  paused = false;
  showStatus();
}

// ---------------------------------------------------------------- main loop
function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  if (video.currentTime === lastVideoTime) return;     // no new frame yet
  lastVideoTime = video.currentTime;

  const nowMs = performance.now(), now = nowMs / 1000;
  fps = 0.9 * fps + 0.1 * (1000 / Math.max(nowMs - lastFrameT, 1));
  lastFrameT = nowMs;

  const result = landmarker.detectForVideo(video, nowMs);
  // mirror x so directions match the selfie view (like cv2.flip in the desktop app)
  const hand = result.landmarks?.[0]?.map((p) => ({ x: 1 - p.x, y: p.y }));
  pose = poseFilter.update(hand ? g.classify(hand) : g.NONE);

  if (pose !== prevPose) {
    smoother.reset(); scroller.reset(); swiper.reset(); stepper.reset();
    prevPose = pose;
    showStatus();
  }

  if (fistHold.update(pose === g.FIST, now)) setPaused(!paused, true);
  if (okHold.update(pose === g.OK && !paused, now)) doOk();

  if (hand && !paused) {
    const palm = g.palmSize(hand);
    if (pose === g.POINT) {
      const [x, y] = smoother.update(hand[8].x, hand[8].y);
      const [, down] = scroller.update(x, y, palm);
      if (down) doScroll(down);
    } else if (pose === g.PALM) {
      const [x, y] = smoother.update(hand[9].x, hand[9].y);
      const dir = swiper.update(x, y, palm, now);
      if (dir) doSwipe(dir);
    } else if (pose === g.TWO) {
      const [, y] = smoother.update(hand[12].x, hand[12].y);
      const steps = stepper.update(y, palm);
      if (steps) doVolume(Math.max(-3, Math.min(3, steps)));
    }
  }

  draw(hand, Math.max(fistHold.progress(now), okHold.progress(now)));
  $("fps").textContent = `${Math.round(fps)} fps`;
}

function draw(hand, progress) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (hand) {
    ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.strokeStyle = "#FFFFFF";
    ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(hand[a].x * w, hand[a].y * h); ctx.lineTo(hand[b].x * w, hand[b].y * h); }
    ctx.stroke();
    ctx.fillStyle = paused ? "#8C93A8" : "#FFC531";
    for (const p of hand) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2); ctx.fill(); }
  }
  $("hold").classList.toggle("show", progress > 0);
  $("hold").firstElementChild.style.width = progress * 100 + "%";
}

// ---------------------------------------------------------------- wiring
$("start").onclick = start;
$("stop").onclick = stop;
$("pause").onclick = () => setPaused(!paused);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && running && e.target === document.body) { e.preventDefault(); setPaused(!paused); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && running) stop(); });
setVolume(50);
showStatus();
