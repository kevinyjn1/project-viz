// AmbientMNIST, under the hood.
// Every number drawn here comes from data/*.json, which tools/ambient-mnist/precompute.py
// produces by replaying the device's integer arithmetic (SONIC libfixed + libdnn) on the
// weights committed in apps/my_mnist/src/headers. The few numbers worked out in this file
// (single products, partial sums) use the same integer rules and are checked against the JSON.
import { createStepper, css } from '../assets/stepper.js';

// ---------------------------------------------------------------- fixed point (libfixed)
// fixed = 16-bit signed int, F_N = 5 fraction bits: real value = integer / 32.
const ONE = 32;
const wrap16 = v => ((((v + 32768) % 65536) + 65536) % 65536) - 32768;
const fmul = (a, b) => wrap16(wrap16(a * b) + 16) >> 5;   // f_mul: (a*b + F_K) >> F_N
const fadd = (a, b) => wrap16(a + b);
const real = v => (v == null ? '–' : String(v / ONE));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmt = n => n.toLocaleString('en-US');

// ---------------------------------------------------------------- data
let WT = null;                 // weights.json
const RUNS = {};               // run-zero.json, run-one.json
let RUNKEY = 'zero';
const R = () => RUNS[RUNKEY];
const STAT = {};               // per run: max |value| per array, counts

function maxAbs(a) { let m = 0; for (const v of a) { const t = Math.abs(v); if (t > m) m = t; } return m || 1; }
function prepRun(run) {
  const s = {};
  for (const k of ['conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2', 'fc1', 'relu3', 'fc2_acc', 'logits']) s[k] = maxAbs(run[k]);
  s.neg1 = run.conv1.filter(v => v < 0).length;
  s.neg2 = run.conv2.filter(v => v < 0).length;
  s.neg3 = run.fc1.filter(v => v < 0).length;
  return s;
}
const sc = k => STAT[RUNKEY][k];
let WS = {};                   // weight scales

// value getters that capture the run they were made for (so switching digits cross-fades)
function mapV(key, ch, H, Wd = H) { const arr = R()[key], off = ch * H * Wd; return (r, c) => arr[off + r * Wd + c]; }
function wV(key, off, cols) { const arr = WT[key]; return (r, c) => arr[off + r * cols + c]; }

// ---------------------------------------------------------------- exact re-computation of single cells
function conv1Detail(ch, i, j, run = R()) {
  const x = run.input, w = WT.conv1_w; const prods = []; let acc = 0;
  for (let l = 0; l < 5; l++) for (let n = 0; n < 5; n++) {
    const wv = w[ch * 25 + l * 5 + n], xv = x[(i + l) * 28 + j + n], p = fmul(wv, xv);
    prods.push({ wv, xv, p }); acc = (l === 0 && n === 0) ? p : fadd(p, acc);
  }
  const b = WT.conv1_b[ch], out = fadd(acc, b);
  return { prods, acc, b, out, ok: out === run.conv1[ch * 576 + i * 24 + j] };
}
function conv2Detail(f, i, j, run = R()) {
  const x = run.pool1, w = WT.conv2_w; let acc = 0, first = true; const per = [];
  for (let k = 0; k < 16; k++) {
    let part = 0;
    for (let l = 0; l < 5; l++) for (let n = 0; n < 5; n++) {
      const p = fmul(w[f * 400 + k * 25 + l * 5 + n], x[k * 144 + (i + l) * 12 + j + n]);
      part += p; acc = first ? p : fadd(p, acc); first = false;
    }
    per.push(part);
  }
  const b = WT.conv2_b[f], out = fadd(acc, b);
  return { per, acc, b, out, ok: out === run.conv2[f * 64 + i * 8 + j] };
}
function fcDetail(Wkey, rows, cols, xin, k, bias) {
  const w = WT[Wkey]; let acc = 0; const prods = [];
  for (let j = 0; j < cols; j++) { const p = fmul(w[k * cols + j], xin[j]); prods.push(p); acc = j === 0 ? p : fadd(p, acc); }
  return { acc, prods, out: bias == null ? acc : fadd(acc, bias) };
}
// partial sums of one conv1 filter, in the device's order: one weight at a time over the whole map
function conv1Partials(ch, run = R()) {
  const P = [null]; const x = run.input, w = WT.conv1_w;
  for (let p = 0; p < 25; p++) {
    const l = (p / 5) | 0, n = p % 5, wv = w[ch * 25 + p], prev = P[p], cur = new Int32Array(576);
    for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) {
      const q = fmul(wv, x[(i + l) * 28 + j + n]);
      cur[i * 24 + j] = p === 0 ? q : fadd(q, prev[i * 24 + j]);
    }
    P.push(cur);
  }
  let ok = true;
  for (let t = 0; t < 576; t++) if (fadd(P[25][t], WT.conv1_b[ch]) !== run.conv1[ch * 576 + t]) ok = false;
  if (!ok) console.warn('AmbientMNIST: partial sums disagree with the precomputed conv1 map', ch);
  return P;
}
function argmaxPos(arr, off, n) { let b = -1e9, bi = 0; for (let t = 0; t < n; t++) if (arr[off + t] > b) { b = arr[off + t]; bi = t; } return bi; }

// ---------------------------------------------------------------- colors
let COL = {}, LUT = {}, FONT = 'sans-serif', MONO = 'monospace';
function parseColor(s) {
  s = (s || '').trim();
  if (s[0] === '#') {
    if (s.length === 4) s = '#' + s[1] + s[1] + s[2] + s[2] + s[3] + s[3];
    const n = parseInt(s.slice(1, 7), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = s.match(/(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)/);
  return m ? [+m[1], +m[2], +m[3]] : [128, 128, 128];
}
const mixc = (a, b, t) => [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * t));
const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;
const lum = c => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
function readColors() {
  for (const n of ['bg', 'surface', 'ink', 'muted', 'line', 'accent', 'flow', 'stage', 'neg', 'pos', 'zero', 'empty', 'cell-line', 'frame', 'good', 'bad'])
    COL[n] = parseColor(css('--' + n));
  FONT = css('--font-body') || FONT; MONO = css('--font-mono') || MONO;
  const div = [], divT = [], img = [], imgT = [];
  for (let i = 0; i <= 256; i++) {
    const t = i / 128 - 1, e = Math.pow(Math.abs(t), 0.72);
    const c = t < 0 ? mixc(COL.zero, COL.neg, e) : mixc(COL.zero, COL.pos, e);
    div.push(rgb(c)); divT.push(lum(c) > 0.5 ? '#111' : '#fff');
  }
  for (let i = 0; i <= 128; i++) { const c = mixc(COL.zero, COL.ink, i / 128); img.push(rgb(c)); imgT.push(lum(c) > 0.5 ? '#111' : '#fff'); }
  LUT = { div, divT, img, imgT, empty: rgb(COL.empty) };
}
function cellColor(pal, v, scale) {
  if (v == null || Number.isNaN(v)) return [LUT.empty, rgb(COL.muted)];
  if (pal === 'img') { const i = Math.round(clamp(v / scale, 0, 1) * 128); return [LUT.img[i], LUT.imgT[i]]; }
  const i = Math.round((clamp(v / scale, -1, 1) + 1) * 128); return [LUT.div[i], LUT.divT[i]];
}
function cellFill(pal, v, scale) {   // same as cellColor()[0], without allocating (used per cell per frame)
  if (v == null || Number.isNaN(v)) return LUT.empty;
  if (pal === 'img') return LUT.img[Math.round(clamp(v / scale, 0, 1) * 128)];
  return LUT.div[Math.round((clamp(v / scale, -1, 1) + 1) * 128)];
}
const C = n => rgb(COL[n] || COL.accent);

// ---------------------------------------------------------------- stage, scene and tweening
let CTX, STAGE, CANVAS, G, FIG, CARD, BADGE;
let DPR = 1, SW = 0, SH = 0;
const SHEETS = new Map(), HLS = new Map();
let LINKS = [], SPEC = null, TICK = null, DIM = 1;
const CAM = { cur: { yaw: 0, pitch: 0, zoom: 10, cx: 0, cy: 0, cz: 0, ax: 0, ay: 0 }, s0: null, tgt: null };
let UYAW = 0, UPITCH = 0, ORBIT = false;
let T0 = 0, DUR = 850, TWEENING = false;
let HOVER = null, DRAWN = [], DRAG = null, raf = 0;
const DATA = { ch: 0, f: 0 };   // shared between steps

const narrow = () => matchMedia('(max-width:820px)').matches;
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// Free stage areas the picture may use. With the card showing there are two choices
// (beside it or below it); camFor picks whichever lets the picture be larger.
function availRects() {
  const box = (x0, y0, x1, y1) => ({ x: x0, y: y0, w: Math.max(100, x1 - x0), h: Math.max(100, y1 - y0) });
  const x0 = 12, y0 = 14, x1 = SW - 12, y1 = SH - 50;
  if (!CARD.classList.contains('show')) return [box(x0, y0, x1, y1)];
  const below = box(x0, CARD.offsetTop + CARD.offsetHeight + 10, x1, y1);
  if (narrow()) return [below];
  return [box(x0, y0, SW - CARD.offsetWidth - 30, y1), below];
}
function rot(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return (dx, dy, dz) => { const x1 = dx * cy + dz * sy, z1 = -dx * sy + dz * cy; return [x1, dy * cp - z1 * sp, dy * sp + z1 * cp]; };
}
function camFor(cam) {
  const b = cam.box, [x0, y0, x1, y1] = b, z0 = b[4] || 0, z1 = b[5] || 0;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, cz = (z0 + z1) / 2;
  const f = rot(cam.yaw || 0, cam.pitch || 0);
  let ex = 1e-3, ey = 1e-3;
  for (const X of [x0, x1]) for (const Y of [y0, y1]) for (const Z of [z0, z1]) {
    const p = f(X - cx, Y - cy, Z - cz); ex = Math.max(ex, Math.abs(p[0])); ey = Math.max(ey, Math.abs(p[1]));
  }
  let av = null, zoom = -1;
  for (const r of availRects()) { const z = Math.min(r.w / (2 * ex), r.h / (2 * ey), cam.maxZoom || 46); if (z > zoom) { zoom = z; av = r; } }
  return { yaw: cam.yaw || 0, pitch: cam.pitch || 0, zoom, cx, cy, cz, ax: av.x + av.w / 2, ay: av.y + av.h / 2 };
}
function proj() {
  const c = CAM.cur, f = rot(c.yaw + UYAW, c.pitch + UPITCH), z = c.zoom;
  return (x, y, zz) => { const p = f(x - c.cx, y - c.cy, zz - c.cz); return [c.ax + p[0] * z, c.ay + p[1] * z, p[2]]; };
}

// A scene lists the sheets (2-D grids of cells), highlights and links that should be on stage.
// Sheets keep their id from step to step, so they glide to their new place instead of popping.
function scene(spec, { snap = false } = {}) {
  SPEC = spec;
  const now = performance.now();
  const seen = new Set();
  for (const s of spec.sheets || []) {
    seen.add(s.id);
    const tgt = { x: s.x, y: s.y, z: s.z || 0, sx: s.sx ?? s.s ?? 1, sy: s.sy ?? s.s ?? 1, a: s.a ?? 1 };
    let o = SHEETS.get(s.id);
    if (!o) {
      o = { id: s.id, cur: s.from ? { ...tgt, ...s.from } : { ...tgt, a: 0 } };
      SHEETS.set(s.id, o);
    } else if (o.key !== s.key && o.cur.a > 0.05 && o.rows === s.rows && o.cols === s.cols) {
      o.v0 = o.v; o.mix = 0;
    } else { o.v0 = null; }
    Object.assign(o, { rows: s.rows, cols: s.cols, v: s.v, key: s.key, pal: s.pal || 'div', scale: s.scale || 1,
      nums: !!s.nums, label: s.label || null, labelTop: !!s.labelTop, info: s.info || null, onHover: s.onHover || null });
    o.tgt = tgt; o.s0 = { ...o.cur };
  }
  for (const [, o] of SHEETS) if (!seen.has(o.id)) { o.tgt = { ...o.cur, a: 0 }; o.s0 = { ...o.cur }; o.info = null; o.onHover = null; o.v0 = null; o.label = null; }
  const hseen = new Set();
  for (const h of spec.hls || []) {
    hseen.add(h.id);
    const tgt = { r: h.r, c: h.c, h: h.h || 1, w: h.w || 1, a: h.a ?? 1 };
    let o = HLS.get(h.id);
    if (!o || o.sheet !== h.sheet) { o = { id: h.id, cur: { ...tgt, a: 0 } }; HLS.set(h.id, o); }
    Object.assign(o, { sheet: h.sheet, color: h.color || 'accent', fill: h.fill ?? 0.12, lw: h.lw || 2 });
    o.tgt = tgt; o.s0 = { ...o.cur };
  }
  for (const [, o] of HLS) if (!hseen.has(o.id)) { o.tgt = { ...o.cur, a: 0 }; o.s0 = { ...o.cur }; }
  LINKS = spec.links || [];
  ORBIT = !!spec.orbit;
  CANVAS.classList.toggle('orbit', ORBIT);
  CAM.s0 = { ...CAM.cur }; CAM.tgt = camFor(spec.cam);
  T0 = now; TWEENING = true;
  if (snap || CTX.reducedMotion) finishTween();
  kick();
}
function finishTween() {
  for (const [id, o] of SHEETS) { o.cur = { ...o.tgt }; o.v0 = null; if (o.tgt.a <= 0) SHEETS.delete(id); }
  for (const [id, o] of HLS) { o.cur = { ...o.tgt }; if (o.tgt.a <= 0) HLS.delete(id); }
  if (CAM.tgt) CAM.cur = { ...CAM.tgt };
  TWEENING = false;
}
function stepTween(now) {
  if (!TWEENING) return false;
  const t = DUR ? Math.min(1, (now - T0) / DUR) : 1;
  if (t >= 1) { finishTween(); return false; }
  const e = ease(t);
  const lerpObj = (o) => { for (const k in o.tgt) o.cur[k] = o.s0[k] + (o.tgt[k] - o.s0[k]) * e; };
  for (const [, o] of SHEETS) { lerpObj(o); if (o.v0) o.mix = e; }
  for (const [, o] of HLS) lerpObj(o);
  for (const k in CAM.tgt) CAM.cur[k] = CAM.s0[k] + (CAM.tgt[k] - CAM.s0[k]) * e;
  return true;
}
// move a highlight right away (used by animations and hover)
function setHl(id, props) {
  const o = HLS.get(id); if (!o) return;
  Object.assign(o.cur, props); Object.assign(o.tgt, props); Object.assign(o.s0, props);
}
function kick() { if (!raf) raf = requestAnimationFrame(frame); }
function frame(now) {
  raf = 0;
  let busy = stepTween(now);
  if (TICK) { try { if (TICK(now) !== false) busy = true; } catch (e) { console.error(e); TICK = null; } }
  draw();
  if (busy) kick();
}

// ---------------------------------------------------------------- drawing
function sheetVal(o, r, c) {
  const v1 = o.v(r, c);
  if (o.v0 && o.mix < 1) { const v0 = o.v0(r, c); if (v0 == null || v1 == null) return o.mix < 0.5 ? v0 : v1; return v0 + (v1 - v0) * o.mix; }
  return v1;
}
function affine(o, P) {
  const { x, y, z, sx, sy } = o.cur;
  const p0 = P(x, y, z), p1 = P(x + sx, y, z), p2 = P(x, y + sy, z);
  return { ox: p0[0], oy: p0[1], ax: p1[0] - p0[0], ay: p1[1] - p0[1], bx: p2[0] - p0[0], by: p2[1] - p0[1] };
}
const pt = (A, r, c) => [A.ox + c * A.ax + r * A.bx, A.oy + c * A.ay + r * A.by];
function quad(A, r, c, h, w) { return [pt(A, r, c), pt(A, r, c + w), pt(A, r + h, c + w), pt(A, r + h, c)]; }
function pathQuad(q) { G.beginPath(); G.moveTo(q[0][0], q[0][1]); for (let i = 1; i < 4; i++) G.lineTo(q[i][0], q[i][1]); G.closePath(); }

function draw() {
  if (!G) return;
  G.setTransform(1, 0, 0, 1, 0, 0);
  G.clearRect(0, 0, CANVAS.width, CANVAS.height);
  const P = proj();
  const list = [];
  for (const [, o] of SHEETS) {
    if (o.cur.a < 0.01) continue;
    const A = affine(o, P);
    const d = P(o.cur.x + o.cur.sx * o.cols / 2, o.cur.y + o.cur.sy * o.rows / 2, o.cur.z)[2];
    list.push({ o, A, d });
  }
  list.sort((a, b) => a.d - b.d);
  DRAWN = list;
  const front = Math.abs(Math.cos(CAM.cur.yaw + UYAW)) > 0.97 && Math.abs(Math.cos(CAM.cur.pitch + UPITCH)) > 0.97;
  for (const it of list) {
    drawSheet(it, front);
    for (const [, h] of HLS) if (h.sheet === it.o.id) drawHl(h, it);
  }
  drawLinks();
  drawHover();
  drawLabels(list);
  G.globalAlpha = 1;
}
function drawSheet({ o, A }, front) {
  const a = o.cur.a * DIM; if (a < 0.01) return;
  const cw = Math.hypot(A.ax, A.ay), chh = Math.hypot(A.bx, A.by), cpx = Math.min(cw, chh);
  G.globalAlpha = a;
  // plate behind the cells
  G.setTransform(DPR, 0, 0, DPR, 0, 0);
  const mx = 2 / Math.max(cw, 0.01), my = 2 / Math.max(chh, 0.01);
  const plate = [pt(A, -my, -mx), pt(A, -my, o.cols + mx), pt(A, o.rows + my, o.cols + mx), pt(A, o.rows + my, -mx)];
  pathQuad(plate); G.fillStyle = C('cell-line'); G.fill();
  if (Math.abs(A.ax * A.by - A.ay * A.bx) < 1e-6) return;
  const gx = cw > 6 ? 0.07 : cw > 3 ? 0.04 : 0, gy = chh > 6 ? 0.07 : chh > 3 ? 0.04 : 0;
  G.setTransform(DPR * A.ax, DPR * A.ay, DPR * A.bx, DPR * A.by, DPR * A.ox, DPR * A.oy);
  for (let r = 0; r < o.rows; r++) for (let c = 0; c < o.cols; c++) {
    G.fillStyle = cellFill(o.pal, sheetVal(o, r, c), o.scale);
    G.fillRect(c + gx, r + gy, 1 - 2 * gx + (gx ? 0 : 0.02), 1 - 2 * gy + (gy ? 0 : 0.02));
  }
  if (o.nums && front && cpx >= 13 && a > 0.5) {
    G.setTransform(DPR, 0, 0, DPR, 0, 0);
    const fs = Math.min(12, cpx * 0.4);
    G.font = `${fs}px ${MONO}`; G.textAlign = 'center'; G.textBaseline = 'middle';
    for (let r = 0; r < o.rows; r++) for (let c = 0; c < o.cols; c++) {
      const v = o.v(r, c); if (v == null) continue;
      const [x, y] = pt(A, r + 0.5, c + 0.5);
      G.fillStyle = cellColor(o.pal, v, o.scale)[1];
      G.fillText(String(v), x, y + 0.5);
    }
  }
}
function drawHl(h, it) {
  const a = h.cur.a * DIM * Math.min(1, it.o.cur.a * 1.5); if (a < 0.02) return;
  G.setTransform(DPR, 0, 0, DPR, 0, 0); G.globalAlpha = a;
  const q = quad(it.A, h.cur.r, h.cur.c, h.cur.h, h.cur.w);
  pathQuad(q);
  if (h.fill) { G.globalAlpha = a * h.fill; G.fillStyle = C(h.color); G.fill(); G.globalAlpha = a; }
  G.lineWidth = h.lw; G.strokeStyle = C(h.color); G.lineJoin = 'round'; G.stroke();
}
function hlQuad(id) {
  const h = HLS.get(id); if (!h) return null;
  const it = DRAWN.find(d => d.o.id === h.sheet); if (!it) return null;
  return { q: quad(it.A, h.cur.r, h.cur.c, h.cur.h, h.cur.w), a: h.cur.a * it.o.cur.a };
}
function drawLinks() {
  G.setTransform(DPR, 0, 0, DPR, 0, 0);
  for (const L of LINKS) {
    const A = hlQuad(L.a), B = hlQuad(L.b); if (!A || !B) continue;
    const a = Math.min(A.a, B.a) * DIM * (L.alpha ?? 0.7); if (a < 0.02) continue;
    G.globalAlpha = a; G.strokeStyle = C(L.color || 'accent'); G.lineWidth = 1.2; G.setLineDash(L.dash ? [4, 4] : []);
    G.beginPath();
    if (L.cone) for (let i = 0; i < 4; i++) { G.moveTo(A.q[i][0], A.q[i][1]); G.lineTo(B.q[i][0], B.q[i][1]); }
    else {
      const ca = A.q.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4], [0, 0]);
      const cb = B.q.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4], [0, 0]);
      G.moveTo(ca[0], ca[1]); G.lineTo(cb[0], cb[1]);
    }
    G.stroke(); G.setLineDash([]);
  }
}
function drawHover() {
  if (!HOVER) return;
  const it = DRAWN.find(d => d.o === HOVER.o); if (!it) return;
  G.setTransform(DPR, 0, 0, DPR, 0, 0); G.globalAlpha = 1;
  for (const s of HOVER.src || []) {
    const si = DRAWN.find(d => d.o.id === s.sheet); if (!si || si.o.cur.a < 0.2) continue;
    pathQuad(quad(si.A, s.r, s.c, s.h || 1, s.w || 1)); G.lineWidth = 2; G.strokeStyle = C('flow'); G.stroke();
  }
  pathQuad(quad(it.A, HOVER.r, HOVER.c, 1, 1)); G.lineWidth = 2; G.strokeStyle = C('ink'); G.stroke();
}
function drawLabels(list) {
  G.setTransform(DPR, 0, 0, DPR, 0, 0);
  G.font = `500 12px ${FONT}`; G.textAlign = 'left'; G.textBaseline = 'alphabetic';
  for (const { o, A } of list) {
    if (!o.label || o.cur.a < 0.25) continue;
    const txt = typeof o.label === 'function' ? o.label() : o.label;
    G.globalAlpha = Math.min(1, o.cur.a) * DIM; G.fillStyle = C('muted');
    if (o.labelTop) { const [x, y] = pt(A, 0, 0); G.fillText(txt, x, y - 7); }
    else { const [x, y] = pt(A, o.rows, 0); G.fillText(txt, x, y + 16); }
  }
}

// ---------------------------------------------------------------- hover, tap, drag
function stageXY(e) { const r = STAGE.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
function hitTest(x, y) {
  for (let k = DRAWN.length - 1; k >= 0; k--) {
    const { o, A } = DRAWN[k];
    if (!o.info || o.cur.a < 0.3) continue;
    const det = A.ax * A.by - A.ay * A.bx; if (Math.abs(det) < 1e-6) continue;
    const dx = x - A.ox, dy = y - A.oy;
    const c = (dx * A.by - dy * A.bx) / det, r = (A.ax * dy - A.ay * dx) / det;
    if (c >= 0 && r >= 0 && c < o.cols && r < o.rows) return { o, r: Math.floor(r), c: Math.floor(c) };
  }
  return null;
}
function hoverAt(e) {
  const [x, y] = stageXY(e);
  const h = hitTest(x, y);
  if (!h) { if (HOVER) { HOVER = null; CTX.tip.hide(); kick(); } return; }
  const res = h.o.info(h.r, h.c) || {};
  HOVER = { o: h.o, r: h.r, c: h.c, src: res.src };
  if (h.o.onHover) h.o.onHover(h.r, h.c);
  if (res.html) CTX.tip.show(x, y, res.html); else CTX.tip.hide();
  kick();
}
function bindPointer() {
  CANVAS.addEventListener('pointerdown', e => {
    DRAG = { x: e.clientX, y: e.clientY, yaw: UYAW, pitch: UPITCH, moved: false };
    if (e.pointerType !== 'mouse') hoverAt(e);
  });
  CANVAS.addEventListener('pointermove', e => {
    if (DRAG && (e.buttons || e.pointerType !== 'mouse')) {
      const dx = e.clientX - DRAG.x, dy = e.clientY - DRAG.y;
      if (!DRAG.moved && Math.hypot(dx, dy) < 6) return;
      if (ORBIT) {
        if (!DRAG.moved) { try { CANVAS.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } }
        DRAG.moved = true; CANVAS.classList.add('dragging');
        UYAW = clamp(DRAG.yaw + dx * 0.006, -0.75, 0.75);
        UPITCH = clamp(DRAG.pitch - dy * 0.004, -0.45, 0.45);
        HOVER = null; CTX.tip.hide(); kick(); return;
      }
    }
    if (e.pointerType === 'mouse') hoverAt(e);
  });
  const end = () => { DRAG = null; CANVAS.classList.remove('dragging'); };
  CANVAS.addEventListener('pointerup', end);
  CANVAS.addEventListener('pointercancel', end);
  CANVAS.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && !DRAG) { HOVER = null; CTX.tip.hide(); kick(); } });
}

// ---------------------------------------------------------------- card, figure, controls
function setCard(html) {
  if (!html) { CARD.classList.remove('show'); CARD.innerHTML = ''; CARD.setAttribute('aria-hidden', 'true'); return; }
  CARD.innerHTML = html; CARD.classList.add('show'); CARD.removeAttribute('aria-hidden');
}
function setFig(html) {
  if (!html) { FIG.classList.remove('show'); FIG.innerHTML = ''; return null; }
  FIG.innerHTML = html; FIG.classList.add('show'); return FIG;
}
function badge(text, cls) {
  if (!text) { BADGE.classList.remove('show'); return; }
  BADGE.innerHTML = text; BADGE.className = 'am-badge show ' + (cls || '');
}
function button(label, onClick, title) {
  const b = document.createElement('button'); b.type = 'button'; b.className = 'am-btn'; b.textContent = label;
  if (title) b.title = title; b.onclick = onClick; CTX.controls.append(b); return b;
}
function runControl() {
  if (!RUNS.zero || !RUNS.one) return;
  const seg = document.createElement('div'); seg.className = 'am-seg'; seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Which input image');
  for (const [k, lab, tip] of [['zero', 'Digit 0 · demo', 'MNIST test image #3, the image in the README demo'],
    ['one', 'Digit 1 · firmware', 'MNIST test image #5, the image compiled into the committed firmware']]) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = lab; b.title = tip;
    b.className = k === RUNKEY ? 'on' : ''; b.setAttribute('aria-pressed', String(k === RUNKEY));
    b.onclick = () => { if (RUNKEY === k) return; RUNKEY = k; rerun(); };
    seg.append(b);
  }
  CTX.controls.append(seg);
}
function chanControl(name, get, set, n = 16) {
  const wrap = document.createElement('div'); wrap.className = 'am-seg';
  const prev = document.createElement('button'), lab = document.createElement('button'), next = document.createElement('button');
  prev.type = lab.type = next.type = 'button';
  prev.textContent = '◀'; next.textContent = '▶'; prev.title = 'Previous ' + name; next.title = 'Next ' + name;
  lab.textContent = `${name} ${get()}`; lab.disabled = true; lab.style.color = 'var(--ink)';
  prev.onclick = () => { set((get() + n - 1) % n); rerun(); };
  next.onclick = () => { set((get() + 1) % n); rerun(); };
  wrap.append(prev, lab, next); CTX.controls.append(wrap);
}
function rerun() {
  const s = CTX.step; if (!s) return;
  if (s.leave) s.leave(CTX);
  CTX.controls.innerHTML = '';
  CTX.tip.hide(); HOVER = null;
  s.enter(CTX);
}

// ---------------------------------------------------------------- sheet makers
const OBL = { yaw: -0.55, pitch: -0.32 };
const FRONT = { yaw: 0, pitch: 0 };
const tipHead = s => `<b>${s}</b><br>`;

function sInput(o = {}) {
  const v = mapV('input', 0, 28);
  return { id: 'in', rows: 28, cols: 28, v, key: RUNKEY + ':in', pal: 'img', scale: 32, x: 0, y: 0, s: 1,
    info: (r, c) => ({ html: `${tipHead(`input · row ${r}, col ${c}`)}stored ${v(r, c)} → ${v(r, c)}/32 = ${real(v(r, c))}` }), ...o };
}
function infoConv1(ch, key) {
  return (r, c) => {
    const d = conv1Detail(ch, r, c);
    if (key === 'relu1') return { html: `${tipHead(`after ReLU · map ${ch} · (${r}, ${c})`)}conv1 gave ${d.out} → relu → ${Math.max(0, d.out)} (${real(Math.max(0, d.out))})`, src: [{ sheet: 'in', r, c, h: 5, w: 5 }] };
    return { html: `${tipHead(`conv1 · map ${ch} · (${r}, ${c})`)}Σ 25 products = ${d.acc}<br>+ bias ${d.b} = ${d.out}<br>real value ${real(d.out)}`, src: [{ sheet: 'in', r, c, h: 5, w: 5 }] };
  };
}
function sConv1(ch, key, o = {}) {
  return { id: `c1_${ch}`, rows: 24, cols: 24, v: mapV(key, ch, 24), key: `${RUNKEY}:${key}:${ch}`, pal: 'div', scale: sc('conv1'),
    info: infoConv1(ch, key), ...o };
}
function infoPool(key, srcKey, ch, H, srcSheet) {
  return (r, c) => {
    const run = R(), S = H * 2, src = run[srcKey], off = ch * S * S;
    const four = [src[off + 2 * r * S + 2 * c], src[off + 2 * r * S + 2 * c + 1], src[off + (2 * r + 1) * S + 2 * c], src[off + (2 * r + 1) * S + 2 * c + 1]];
    const v = run[key][ch * H * H + r * H + c];
    return { html: `${tipHead(`max-pool · map ${ch} · (${r}, ${c})`)}max(${four.join(', ')}) = ${v}<br>real value ${real(v)}`, src: srcSheet ? [{ sheet: srcSheet, r: 2 * r, c: 2 * c, h: 2, w: 2 }] : [] };
  };
}
function sPool1(ch, o = {}) {
  return { id: `p1_${ch}`, rows: 12, cols: 12, v: mapV('pool1', ch, 12), key: `${RUNKEY}:pool1:${ch}`, pal: 'div', scale: sc('pool1'),
    info: infoPool('pool1', 'relu1', ch, 12, `c1_${ch}`), ...o };
}
function infoConv2(f, key) {
  return (r, c) => {
    const d = conv2Detail(f, r, c);
    const src = []; for (let k = 0; k < 16; k++) src.push({ sheet: `p1_${k}`, r, c, h: 5, w: 5 });
    if (key === 'relu2') return { html: `${tipHead(`after ReLU · conv2 map ${f} · (${r}, ${c})`)}conv2 gave ${d.out} → ${Math.max(0, d.out)}`, src };
    return { html: `${tipHead(`conv2 · map ${f} · (${r}, ${c})`)}Σ 400 products = ${d.acc}<br>+ bias ${d.b} = ${d.out}<br>real value ${real(d.out)}`, src };
  };
}
function sConv2(f, key, id, o = {}) {
  return { id, rows: 8, cols: 8, v: mapV(key, f, 8), key: `${RUNKEY}:${key}:${f}`, pal: 'div', scale: sc('conv2'), info: infoConv2(f, key), ...o };
}
function sPool2(ch, o = {}) {
  return { id: `p2_${ch}`, rows: 4, cols: 4, v: mapV('pool2', ch, 4), key: `${RUNKEY}:pool2:${ch}`, pal: 'div', scale: sc('pool2'),
    info: infoPool('pool2', 'relu2', ch, 4, `r2_${ch}`), ...o };
}
function sVec(id, key, n, o = {}) {   // column vector n x 1
  const arr = R()[key];
  return { id, rows: n, cols: 1, v: (r) => arr[r], key: `${RUNKEY}:${key}`, pal: 'div', scale: sc(key) || 1, ...o };
}
function sFlatSeg(k, o = {}) {        // 1 x 16 piece of the flattened 256-vector
  const arr = R().pool2;
  return { id: `fl_${k}`, rows: 1, cols: 16, v: (r, c) => arr[k * 16 + c], key: `${RUNKEY}:flat:${k}`, pal: 'div', scale: sc('pool2'),
    info: (r, c) => { const j = k * 16 + c; return { html: `${tipHead(`flat[${j}]`)}= pool map ${k}, row ${(c / 4) | 0}, col ${c % 4}<br>value ${arr[j]} (${real(arr[j])})`, src: [{ sheet: `p2_${k}`, r: (c / 4) | 0, c: c % 4 }] }; }, ...o };
}

// overview layout: the whole network as stacks of maps
function overviewSheets(a = 1) {
  const out = [];
  out.push(sInput({ a, label: 'input 28×28' }));
  for (let k = 0; k < 16; k++) out.push(sConv1(k, 'conv1', { x: 36, y: 2, z: -k * 1.4, a, label: k === 0 ? 'conv1 · 16 maps × 24×24' : null }));
  for (let k = 0; k < 16; k++) out.push(sPool1(k, { x: 80, y: 8, z: -k * 1.4, a, label: k === 0 ? 'pool · 16 × 12×12' : null }));
  for (let k = 0; k < 16; k++) out.push(sConv2(k, 'conv2', `c2_${k}`, { x: 106, y: 10, z: -k * 1.0, a, label: k === 0 ? 'conv2 · 16 × 8×8' : null }));
  for (let k = 0; k < 16; k++) out.push(sPool2(k, { x: 126, y: 12, z: -k * 1.0, a, label: k === 0 ? 'pool · 16 × 4×4' : null }));
  out.push(sVec('r3', 'relu3', 32, { x: 142, y: -2, s: 1, a, label: 'fc1 · 32', info: infoFc1 }));
  out.push(sVec('lg', 'logits', 10, { x: 150, y: 9, s: 1, a, label: 'scores · 10', info: infoLogit }));
  return out;
}
const OVERVIEW_CAM = { ...OBL, box: [-2, -4, 154, 34, -21, 0] };
function infoFc1(r) {
  const run = R(), v = run.fc1[r];
  return { html: `${tipHead(`fc1 output ${r}`)}Σ 256 products = ${v} (no bias on the device)<br>after ReLU ${run.relu3[r]} = ${real(run.relu3[r])}` };
}
function infoLogit(r) {
  const run = R();
  return { html: `${tipHead(`score for digit ${r}`)}Σ 32 products = ${run.fc2_acc[r]}<br>+ bias ${WT.fc2_b[r]} = ${run.logits[r]} (${real(run.logits[r])})` };
}

// ---------------------------------------------------------------- figures (SVG / HTML)
function svgEl(w, h, inner) { return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img">${inner}</svg>`; }
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

function figMemory() {
  const fw = WT.firmware_bytes || {};
  const FR = 262144, SR = 8192, px = 560 / FR, x0 = 30;
  const segs = [
    ['program code', fw['.text'] || 0, 'var(--frame)'],
    ['weights + input', Object.values(WT.bytes).reduce((a, b) => a + b, 0), 'var(--accent)'],
    ['feature-map buffers', fw.layer_buffers || 0, 'var(--flow)'],
    ['scratch + commit log', (fw.mat_buffers || 0) + (fw.data_src || 0) + (fw.data_dest || 0) + (fw.data_size || 0), 'var(--good)'],
  ];
  let x = x0, rects = '', legend = '';
  segs.forEach(([name, b, col], i) => {
    rects += `<rect x="${x}" y="70" width="${Math.max(1, b * px)}" height="34" style="fill:${col}"><title>${name}: ${fmt(b)} bytes</title></rect>`;
    legend += `<rect x="${x0 + (i % 2) * 290}" y="${122 + ((i / 2) | 0) * 22}" width="12" height="12" rx="2" style="fill:${col}"/><text x="${x0 + 18 + (i % 2) * 290}" y="${133 + ((i / 2) | 0) * 22}" font-size="13">${name} <tspan class="mono muted" font-size="12">${fmt(b)} B</tspan></text>`;
    x += b * px;
  });
  const fmap = 16 * 24 * 24 * 2;
  return svgEl(620, 330, `
    <text x="${x0}" y="58" font-size="15" font-weight="600">FRAM · 256 KB · keeps its contents with the power off</text>
    <rect x="${x0}" y="70" width="${FR * px}" height="34" rx="3" style="fill:var(--zero);stroke:var(--line)"/>${rects}
    ${legend}
    <text x="${x0}" y="208" font-size="15" font-weight="600">SRAM · 8 KB · forgets everything when power drops</text>
    <rect x="${x0}" y="220" width="${SR * px}" height="34" rx="2" style="fill:var(--bad)"/>
    <text x="${x0 + SR * px + 8}" y="242" font-size="13" class="mono muted">${fmt(SR)} B</text>
    <rect x="${x0}" y="270" width="${fmap * px}" height="20" rx="2" style="fill:none;stroke:var(--flow);stroke-dasharray:4 3"/>
    <text x="${x0 + fmap * px + 8}" y="285" font-size="13">one conv1 output (16×24×24 × 2 B) = <tspan class="mono">${fmt(fmap)} B</tspan></text>
    <text x="${x0}" y="318" font-size="12" class="muted">Sizes read from the compiled firmware (apps/my_mnist/bld/gcc/my_mnist.out). Bars drawn to scale.</text>`);
}
function figFixed(ch) {
  const tr = WT.conv1_w_trained.slice(ch * 25, ch * 25 + 25), q = WT.conv1_w.slice(ch * 25, ch * 25 + 25);
  const lo = -0.4, hi = 0.4, X = v => 50 + (v - lo) / (hi - lo) * 300, Y = v => 230 - (v - lo) / (hi - lo) * 200;
  let stair = '';
  for (let k = Math.ceil(lo * 32); k <= Math.floor(hi * 32); k++) {
    // F_LIT truncates toward zero: values in (k/32, (k+1)/32) map to k for k >= 0, to k+1 for negatives
    const a = k > 0 ? k / 32 : (k - 1) / 32, b = k >= 0 ? (k + 1) / 32 : k / 32;
    if (a < lo || b > hi) continue;
    stair += `<line x1="${X(a)}" y1="${Y(k / 32)}" x2="${X(b)}" y2="${Y(k / 32)}" style="stroke:var(--frame);stroke-width:1.5"/>`;
  }
  let dots = '';
  tr.forEach((t, i) => { dots += `<circle cx="${X(t)}" cy="${Y(q[i] / 32)}" r="3.4" style="fill:var(--accent)"><title>trained ${t} → ×32 = ${(t * 32).toFixed(3)} → stored ${q[i]} = ${real(q[i])}</title></circle>`; });
  // a worked product from the conv1 window used on the next steps
  const pos = DATA.c1pos || { i: 0, j: 0 }, d = conv1Detail(ch, pos.i, pos.j);
  let best = d.prods[0]; for (const p of d.prods) if (Math.abs(p.wv * p.xv) > Math.abs(best.wv * best.xv)) best = p;
  const a = best.wv, b = best.xv, ab = a * b, z = WT.zero_weights;
  return svgEl(680, 300, `
    <text x="50" y="20" font-size="14" font-weight="600">Storing filter ${ch}'s 25 trained weights</text>
    <line x1="${X(lo)}" y1="${Y(0)}" x2="${X(hi)}" y2="${Y(0)}" style="stroke:var(--line)"/>
    <line x1="${X(0)}" y1="${Y(lo)}" x2="${X(0)}" y2="${Y(hi)}" style="stroke:var(--line)"/>
    ${stair}${dots}
    <text x="${X(hi)}" y="${Y(0) + 16}" font-size="11" text-anchor="end" class="muted">trained value</text>
    <text x="${X(0) + 6}" y="${Y(hi) + 4}" font-size="11" class="muted">stored value (steps of 1/32)</text>
    <text x="50" y="258" font-size="12" class="muted">Each dot: a real weight of filter ${ch}. Hover a dot for its numbers.</text>
    <text x="50" y="276" font-size="12" class="muted">Rounded to 0 by truncation: ${z.conv1_w}/400 conv1, ${fmt(z.conv2_w)}/6,400 conv2,</text>
    <text x="50" y="292" font-size="12" class="muted">${fmt(z.fc1_w)}/8,192 fc1 and ${z.fc2_w}/320 fc2 weights.</text>
    <g transform="translate(400,40)">
      <text x="0" y="0" font-size="14" font-weight="600">One multiply, the chip's way</text>
      <text x="0" y="28" font-size="13" class="mono">weight  ${a}  (= ${real(a)})</text>
      <text x="0" y="48" font-size="13" class="mono">pixel   ${b}  (= ${real(b)})</text>
      <text x="0" y="76" font-size="13" class="mono">a × b       = ${ab}</text>
      <text x="0" y="96" font-size="13" class="mono">+ 16        = ${ab + 16}</text>
      <text x="0" y="116" font-size="13" class="mono">&gt;&gt; 5 (÷32)  = <tspan style="fill:var(--accent);font-weight:600">${fmul(a, b)}</tspan></text>
      <text x="0" y="144" font-size="12" class="muted">${fmul(a, b)}/32 = ${real(fmul(a, b))}</text>
      <text x="0" y="162" font-size="12" class="muted">exact: ${real(a)} × ${real(b)} = ${(a * b / 1024).toFixed(5)}</text>
      <text x="0" y="190" font-size="12" class="muted">16-bit integers: −32768 … 32767,</text>
      <text x="0" y="206" font-size="12" class="muted">so values from −1024 to 1023.97.</text>
    </g>`);
}
function figLogits() {
  const run = R(), L = run.logits, m = Math.max(...L.map(Math.abs), 1);
  const x0 = 40, w = 52, mid = 160, hs = 120 / m;
  let g = '';
  L.forEach((v, d) => {
    const x = x0 + d * w, h = Math.abs(v) * hs, y = v >= 0 ? mid - h : mid, win = d === run.prediction;
    g += `<rect x="${x + 8}" y="${y}" width="${w - 16}" height="${Math.max(1, h)}" rx="2" style="fill:var(${v >= 0 ? '--pos' : '--neg'});${win ? 'stroke:var(--accent);stroke-width:3' : ''}"><title>digit ${d}: ${v} = ${real(v)}</title></rect>
      <text x="${x + w / 2}" y="${v >= 0 ? y - 6 : y + h + 15}" font-size="12" text-anchor="middle" class="mono">${v}</text>
      <text x="${x + w / 2}" y="306" font-size="15" text-anchor="middle" font-weight="${win ? 700 : 400}" style="${win ? 'fill:var(--accent)' : ''}">${d}</text>`;
  });
  return svgEl(600, 330, `<line x1="${x0}" y1="${mid}" x2="${x0 + 10 * w}" y2="${mid}" style="stroke:var(--frame)"/>${g}
    <text x="${x0}" y="22" font-size="14" font-weight="600">Scores (stored integers; divide by 32 for the real value)</text>
    <text x="${x0}" y="326" font-size="12" class="muted">Answer: digit ${run.prediction} (score ${L[run.prediction]} = ${real(L[run.prediction])}).</text>`);
}
function figMatch() {
  const run = R(), dev = run.device_logits, fl = run.float_logits, nb = run.notebook_logits ? run.notebook_logits.map(v => Math.round(v * ONE)) : null;
  const am = a => a.indexOf(Math.max(...a));
  const td = (a, d, txt) => `<td class="${am(a) === d ? 'win' : ''}">${txt}</td>`;
  const rows = run.logits.map((v, d) => `<tr><td>${d}</td>${dev ? td(dev, d, dev[d]) : ''}${nb ? td(nb, d, nb[d]) : ''}${td(run.logits, d, v)}${td(fl, d, fl[d].toFixed(2))}</tr>`).join('');
  return `<div class="tbl"><table><thead><tr><th>digit</th>${dev ? '<th>device printout<br>(README)</th>' : ''}${nb ? '<th>notebook emulator<br>(earlier weights) ×32</th>' : ''}<th>this replay<br>(committed weights)</th><th>float math<br>(committed weights)</th></tr></thead><tbody>${rows}</tbody></table>
    <p>${dev ? (run.match ? 'All ten integers match the device.' : 'All columns pick the same digit; the scores differ.') : 'The repo has no device printout for this image.'} The highest score in each column is in color.${nb ? ' Notebook column: the Python emulator\'s printed logits in AmbientMNIST_Script.ipynb, times 32.' : ''} Float column: ordinary decimals, no rounding, fc1 bias included (as in TensorFlow).</p></div>`;
}
// illustrative supply-voltage shape for the power chapter (no real measurements)
function vShape(t) {   // t in [0, 3): three charge/run cycles
  const k = t % 1, cyc = Math.floor(t);
  const start = cyc === 0 ? 0 : 0.30;
  if (k < 0.58) { const u = k / 0.58; return { v: start + (0.85 - start) * (1 - Math.exp(-3.2 * u)) / (1 - Math.exp(-3.2)), on: false }; }
  const u = (k - 0.58) / 0.42; return { v: 0.85 - 0.55 * Math.pow(u, 0.9), on: true };
}
function figPower() {
  const X = t => 60 + t * 170, Y = v => 360 - v * 170;
  let d = '', shade = '';
  for (let i = 0; i <= 600; i++) { const t = i / 200, p = vShape(t); d += (i ? 'L' : 'M') + X(t).toFixed(1) + ' ' + Y(p.v).toFixed(1); }
  for (let c = 0; c < 3; c++) shade += `<rect x="${X(c + 0.58)}" y="${Y(1)}" width="${X(c + 1) - X(c + 0.58)}" height="${Y(0) - Y(1)}" style="fill:var(--good);opacity:.10"/>`;
  return svgEl(640, 400, `
    <g transform="translate(40,30)">
      <circle cx="30" cy="40" r="20" style="fill:var(--accent)"/>
      ${[0, 1, 2, 3, 4, 5, 6, 7].map(i => { const a = i * Math.PI / 4; return `<line x1="${30 + Math.cos(a) * 26}" y1="${40 + Math.sin(a) * 26}" x2="${30 + Math.cos(a) * 34}" y2="${40 + Math.sin(a) * 34}" style="stroke:var(--accent);stroke-width:3;stroke-linecap:round"/>`; }).join('')}
      <polygon points="100,20 200,20 186,64 86,64" style="fill:var(--flow);opacity:.85"/>
      ${[1, 2, 3].map(i => `<line x1="${100 + i * 25}" y1="20" x2="${86 + i * 25}" y2="64" style="stroke:var(--surface);stroke-width:1.5"/>`).join('')}
      <text x="88" y="88" font-size="12">solar panel, 600 mm²</text>
      <line x1="205" y1="42" x2="268" y2="42" style="stroke:var(--frame);stroke-width:2"/>
      <line x1="270" y1="22" x2="270" y2="62" style="stroke:var(--ink);stroke-width:3"/><line x1="282" y1="22" x2="282" y2="62" style="stroke:var(--ink);stroke-width:3"/>
      <text x="236" y="88" font-size="12">storage capacitor</text>
      <line x1="284" y1="42" x2="370" y2="42" style="stroke:var(--frame);stroke-width:2"/>
      <rect id="am-chip" x="372" y="10" width="170" height="64" rx="6" style="fill:var(--zero);stroke:var(--frame);stroke-width:2"/>
      <text x="457" y="36" font-size="13" text-anchor="middle" font-weight="600">MSP430FR5994</text>
      <text id="am-chipstate" x="457" y="58" font-size="13" text-anchor="middle" class="mono">off</text>
    </g>
    ${shade}
    <line x1="${X(0)}" y1="${Y(0.85)}" x2="${X(3)}" y2="${Y(0.85)}" style="stroke:var(--good);stroke-dasharray:5 4"/>
    <text x="${X(3) + 4}" y="${Y(0.85) + 4}" font-size="11" class="muted">turn on</text>
    <line x1="${X(0)}" y1="${Y(0.30)}" x2="${X(3)}" y2="${Y(0.30)}" style="stroke:var(--bad);stroke-dasharray:5 4"/>
    <text x="${X(3) + 4}" y="${Y(0.30) + 4}" font-size="11" class="muted">too low</text>
    <line x1="${X(0)}" y1="${Y(0)}" x2="${X(3)}" y2="${Y(0)}" style="stroke:var(--line)"/>
    <line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(1)}" style="stroke:var(--line)"/>
    <text x="${X(0) - 8}" y="${Y(0.5)}" font-size="12" text-anchor="end" class="muted">voltage</text>
    <text x="${X(3)}" y="${Y(0) + 18}" font-size="12" text-anchor="end" class="muted">time →</text>
    <path d="${d}" style="fill:none;stroke:var(--accent);stroke-width:2.2"/>
    <circle id="am-vdot" cx="${X(0)}" cy="${Y(0)}" r="6" style="fill:var(--accent);stroke:var(--surface);stroke-width:2"/>
    <text x="${X(0)}" y="392" font-size="12" class="muted">Illustrative shape, not a measurement. Green bands: the chip has enough energy to compute.</text>`);
}
function figSurvive() {
  const fw = WT.firmware_bytes || {};
  const li = a => a.map(s => `<li>${s}</li>`).join('');
  return `<div class="tbl" style="max-width:680px">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div><div style="font-weight:600;color:var(--bad)">SRAM: lost at power-off</div><ul style="margin:6px 0 0;padding-left:18px">${li(['CPU registers', 'the C stack', 'local loop variables (i, j, the current product)'])}</ul></div>
      <div><div style="font-weight:600;color:var(--good)">FRAM: kept</div><ul style="margin:6px 0 0;padding-left:18px">${li([
        `all weights (${fmt(Object.entries(WT.bytes).filter(([k]) => k !== 'input').reduce((a, [, b]) => a + b, 0))} B)`,
        `the feature-map buffers b1 / b2 (${fmt(fw.layer_buffers || 0)} B for four layer buffers)`,
        'which task is running: <code>curctx</code>, pointing at one of two copies, context_0 / context_1',
        'each task\'s counters <code>CUR_SCRATCH[]</code>: which layer state, which filter, which row and column',
        'the commit list (<code>data_src</code>, <code>data_dest</code>, <code>data_size</code>)'])}</ul></div>
    </div>
    <div style="margin-top:12px;font-weight:600">How a task hands over (libalpaca, used by SONIC)</div>
    <ol style="margin:6px 0 0;padding-left:20px">
      <li><code>write_to_gbuf</code>: write the new counter value into a list instead of overwriting it (e.g. "state = 4").</li>
      <li><code>transition_to</code>: fill in the spare context and point <code>curctx</code> at it: one pointer write, so it either happened or it didn't.</li>
      <li><code>task_prologue</code>: copy the listed values into place. If power dies halfway, the reboot runs the prologue again and copies them again; the result is the same.</li>
    </ol>
    <p>On every boot, <code>main()</code> runs <code>task_prologue</code> and jumps straight to <code>curctx-&gt;task</code>: the task that was running, not the start of the program.</p></div>`;
}
function figTasks() {
  const rows = [
    ['0', 'copy the image into buffer b2', '1 task, 784 copies', 0],
    ['1', 'conv1: task_d_conv', `16 filters × 25 weights = 400 task_dm_conv passes, + 16 bias tasks`, 230400],
    ['2', 'ReLU', 'task_relu over 9,216 values', 0],
    ['3', 'max-pool 2×2', 'task_pool → 16×12×12', 0],
    ['4', 'conv2: task_d_conv', '16 filters × 400 weights = 6,400 task_dm_conv passes, + 16 bias tasks', 409600],
    ['5', 'ReLU', '1,024 values', 0],
    ['6', 'max-pool 2×2', '→ 16×4×4 = 256', 0],
    ['7', 'fc1: task_d_fc', '256 task_dm_mul passes (one per input), no bias', 8192],
    ['8', 'ReLU', '32 values', 0],
    ['9', 'fc2: task_d_fc', '32 task_dm_mul passes + task_dm_add for the bias', 320],
    ['–', 'task_finish', 'pick the largest score, print all 10 over UART', 0],
  ];
  const tot = rows.reduce((a, r) => a + r[3], 0), mx = 409600;
  const tr = rows.map(r => `<tr><td>${r[0]}</td><td style="text-align:left;font-family:var(--font-body)">${r[1]}<div class="muted" style="font-size:11.5px">${r[2]}</div></td><td>${r[3] ? fmt(r[3]) : ''}${r[3] ? `<div style="height:6px;background:var(--accent);border-radius:2px;margin-top:2px;width:${Math.max(2, r[3] / mx * 100)}%;margin-left:auto"></div>` : ''}</td></tr>`).join('');
  return `<div class="tbl" style="max-width:640px"><table><thead><tr><th>state</th><th style="text-align:left">what task_compute starts</th><th>multiplies</th></tr></thead><tbody>${tr}
    <tr><td></td><td style="text-align:left;font-family:var(--font-body);font-weight:600">total</td><td style="font-weight:600">${fmt(tot)}</td></tr></tbody></table>
    <p>Counts follow main.c and SONIC's libdnn loops for this network. Each pass is a separate task, so a blackout costs at most the pass in progress, and inside a pass the saved row/column means only one cell is redone.</p></div>`;
}

// ---------------------------------------------------------------- the steps
function vals() {
  const run = R(), z = RUNKEY === 'zero';
  return {
    img: z ? 'MNIST test image #3, a handwritten <b>0</b>: the image from the README demo'
      : 'MNIST test image #5, a handwritten <b>1</b>: the image compiled into the firmware committed in the repo (headers/input.h)',
    imgcheck: z ? 'We checked: these 784 integers match, one for one, the 28×28 dump the device printed in the README (input1.png).'
      : 'These 784 integers are exactly what the committed firmware binary (my_mnist.out) carries; we read them back out of it to check.',
    pred: String(run.prediction),
    matchtext: z
      ? `<p>The README shows the device's printout for this exact image: <code>0 =&gt; 247</code>, the highest of the ten. Our replay also answers <b>${run.prediction}</b>, but with different scores (${run.logits[0]} for "0").</p>
         <p>The input matches the device's dump digit for digit, and the arithmetic follows SONIC's code line by line, so the difference is the weights: the demo ran on weights from an earlier training run than the ones committed in the repo. The notebook backs this up. Its own Python emulator, run in the session that trained those earlier weights, printed scores that land close to the device's (232 vs 247 for "0", 87 vs 89 for "6"); the small gaps come from the emulator's simpler rounding and its use of the fc1 bias.</p>`
      : `<p>The README only shows a device printout for the demo "0". For this "1", the committed firmware would compute the scores in our middle column (the arithmetic is the same), but no printout of it is in the repo, so we can't compare.</p>`,
  };
}
function fillVals() {
  const V = vals();
  document.querySelectorAll('#step-body [data-v]').forEach(el => { el.innerHTML = V[el.dataset.v] ?? ''; });
}

function bigFilterSheet(ch, o = {}) {
  const w = wV('conv1_w', ch * 25, 5), tr = WT.conv1_w_trained;
  return { id: 'k1big', rows: 5, cols: 5, v: w, key: 'k1:' + ch, pal: 'div', scale: WS.conv1_w, nums: true,
    info: (r, c) => { const q = w(r, c), t = tr[ch * 25 + r * 5 + c]; return { html: `${tipHead(`filter ${ch} · weight (${r}, ${c})`)}trained ${t}<br>× 32 = ${(t * 32).toFixed(3)} → stored ${q}<br>= ${real(q)}` }; }, ...o };
}

function stepOverview() {
  return {
    title: 'A digit reader with no battery',
    body: `<p>AmbientMNIST reads a handwritten digit on a <b>TI MSP430FR5994</b>, a microcontroller clocked at 16 MHz, powered only by a small solar panel. When the light fades, the chip simply turns off, even in the middle of the math, and picks up again later.</p>
      <p>On the right is the whole network the chip runs, left to right: the 28×28 image, two convolution layers (each followed by ReLU and 2×2 max-pooling), then two fully connected layers that end in 10 scores, one per digit.</p>
      <p>Every cell is a real number computed for one image. <span style="color:var(--pos)">Orange</span> is positive, <span style="color:var(--neg)">blue</span> negative. Hover or tap any cell to see its value; drag to turn the view. The switch at the bottom changes the input image.</p>`,
    enter() {
      setFig(null);
      setCard(`<h3>The setup</h3>
        <div class="row"><span>chip</span><span>MSP430FR5994</span></div>
        <div class="row"><span>clock</span><span>16 MHz</span></div>
        <div class="row"><span>memory</span><span>256 KB FRAM, 8 KB SRAM</span></div>
        <div class="row"><span>power</span><span>600 mm² solar panel</span></div>
        <div class="row"><span>framework</span><span>SONIC (CMU)</span></div>
        <div class="row"><span>numbers</span><span>16-bit fixed point</span></div>
        <div class="row"><span>answer here</span><span class="acc">${R().prediction}</span></div>`);
      scene({ cam: OVERVIEW_CAM, sheets: overviewSheets(), orbit: true });
    },
  };
}
function stepMemory() {
  return {
    title: 'Where everything lives',
    body: `<p>The chip has two kinds of memory. <b>SRAM</b> (8 KB) is fast but forgets everything the instant power drops. <b>FRAM</b> (256 KB) is slower but keeps its contents with no power at all, like a tiny SSD you can write one byte at a time.</p>
      <p>One conv1 output alone is 16 × 24 × 24 numbers × 2 bytes = 18,432 bytes, more than twice the whole SRAM. So the weights and every intermediate result live in FRAM.</p>
      <p>That choice is forced by size, but it also turns out to be what lets the network survive a blackout (last chapter).</p>`,
    enter() {
      setCard(null);
      setFig(figMemory());
      scene({ cam: OVERVIEW_CAM, sheets: overviewSheets(0.1) });
    },
  };
}
function stepInput() {
  return {
    title: '28 × 28 numbers',
    body: `<p>The input is <span data-v="img"></span>.</p>
      <p>Each pixel is stored as a whole number from 0 (black) to 32 (white). The real value is that number divided by 32, so 31 means 0.96875. Hover a pixel to see it.</p>
      <p class="note" data-v="imgcheck"></p>`,
    enter() {
      setFig(null);
      const run = R(); let on = 0; for (const v of run.input) if (v > 0) on++;
      setCard(`<h3>Input image</h3>
        <div class="row"><span>pixels</span><span>784</span></div>
        <div class="row"><span>non-zero</span><span>${on}</span></div>
        <div class="row"><span>stored as</span><span>0 … 32</span></div>
        <div class="row"><span>true label</span><span>${run.label}</span></div>
        <p class="muted" style="margin-top:6px">Pixel 0–255 → ÷255 → ×32 → drop the fraction.</p>`);
      scene({ cam: { ...FRONT, box: [-1, -1, 29, 30] }, sheets: [sInput({ nums: true, label: 'input · 28×28' })] });
    },
  };
}
function stepFixed() {
  return {
    title: 'Fixed point: whole numbers standing in for decimals',
    body: `<p>The chip has no hardware for decimals. So every number is a 16-bit integer that quietly means "this ÷ 32" (5 bits after the binary point). The README's "5-bit fixed point" means exactly this: 5 fraction bits inside a 16-bit integer.</p>
      <p>To store a trained weight, multiply it by 32 and chop off the fraction (<code>F_LIT</code> in SONIC's libfixed). Anything smaller than 1/32 = 0.03125 becomes 0.</p>
      <p>Multiplying two such numbers gives "÷ 1024", one scale too many. So <code>f_mul</code> adds 16 and shifts right by 5 bits: divide by 32, rounded to the nearest step. Every multiply in this network does that.</p>`,
    enter() {
      setCard(null);
      ensureC1Pos();
      setFig(figFixed(DATA.ch));
      scene({ cam: { ...FRONT, box: [-1, -1, 29, 30] }, sheets: [sInput({ a: 0.1 })] });
    },
  };
}
function stepFilter() {
  return {
    title: 'A 5×5 filter',
    body: `<p>conv1 has 16 <b>filters</b>. Each one is a 5×5 grid of weights: a small pattern detector. <span style="color:var(--pos)">Orange</span> weights reward a bright pixel in that spot, <span style="color:var(--neg)">blue</span> weights penalize one.</p>
      <p>Hover a weight to see its trained value and the integer actually stored on the chip. Use ◀ ▶ to look at the other filters.</p>
      <p>All 16 filters together are 400 weights, 800 bytes.</p>`,
    enter() {
      setFig(null); setCard(null);
      chanControl('filter', () => DATA.ch, v => { DATA.ch = v; DATA.c1pos = null; });
      const ch = DATA.ch, sheets = [sInput({ label: 'input' }), bigFilterSheet(ch, { x: 32, y: 0, s: 2.4, label: `filter ${ch} · 5×5` })];
      for (let k = 0; k < 16; k++) sheets.push({ ...bigFilterSheet(k), id: `k1_${k}`, nums: false, x: 32 + (k % 8) * 3.3, y: 15 + ((k / 8) | 0) * 3.7, s: 0.55, label: k === 0 ? 'all 16 conv1 filters' : null });
      scene({ cam: { ...FRONT, box: [-1, -2, 59.5, 25] }, sheets, hls: [{ id: 'selk', sheet: `k1_${ch}`, r: -0.15, c: -0.15, h: 5.3, w: 5.3, fill: 0 }] });
    },
  };
}
function ensureC1Pos() {
  if (!DATA.c1pos || DATA.c1posFor !== RUNKEY + ':' + DATA.ch) {
    const t = argmaxPos(R().conv1, DATA.ch * 576, 576);
    DATA.c1pos = { i: (t / 24) | 0, j: t % 24 }; DATA.c1posFor = RUNKEY + ':' + DATA.ch;
  }
  return DATA.c1pos;
}
function cardConv1(ch, i, j) {
  const d = conv1Detail(ch, i, j);
  const cells = d.prods.map(p => `<div title="f_mul(${p.wv}, ${p.xv})">${p.p}<small>${p.wv}·${p.xv}</small></div>`).join('');
  return `<h3>conv1 · map ${ch} · row ${i}, col ${j}</h3>
    <p class="muted">25 products, each f_mul(weight, pixel):</p>
    <div class="am-prods">${cells}</div>
    <div class="eq">Σ ${d.acc} + bias ${d.b} = <b>${d.out}</b></div>
    <div class="row"><span>real value</span><span>${d.out}/32 = ${real(d.out)}</span></div>`;
}
function stepSlide() {
  let sweeping = false;
  return {
    title: 'Slide, multiply, add',
    body: `<p>To get one output number, lay the filter over a 5×5 patch of the image, multiply the 25 overlapping pairs, add them up, then add the filter's <b>bias</b> (one extra number per filter).</p>
      <p>Then slide one pixel over and do it again. A 5-wide filter fits 24 positions across a 28-wide image, so each filter produces a 24×24 <b>feature map</b>.</p>
      <p>Hover any cell of the output map to see exactly which pixels and products made it. Press <b>Sweep</b> to watch the whole map fill in.</p>`,
    enter() {
      setFig(null);
      chanControl('filter', () => DATA.ch, v => { DATA.ch = v; });
      const ch = DATA.ch, pos = ensureC1Pos();
      let sweep = 576;
      const full = mapV('conv1', ch, 24);
      const outV = (r, c) => (r * 24 + c < sweep ? full(r, c) : null);
      const place = (i, j) => { setHl('win', { r: i, c: j }); setHl('cell', { r: i, c: j }); setCard(cardConv1(ch, i, j)); };
      setCard(cardConv1(ch, pos.i, pos.j));
      scene({
        cam: { ...FRONT, box: [-1, -1.5, 68, 30] },
        sheets: [sInput({ label: 'input' }), bigFilterSheet(ch, { x: 31, y: 9.5, s: 1.8, label: `filter ${ch}` }),
          { ...sConv1(ch, 'conv1'), id: 'c1out', v: outV, x: 43, y: 2, label: `conv1 map ${ch} · 24×24`,
            onHover: (r, c) => { if (!sweeping) { DATA.c1pos = { i: r, j: c }; place(r, c); } } }],
        hls: [{ id: 'win', sheet: 'in', r: pos.i, c: pos.j, h: 5, w: 5 }, { id: 'cell', sheet: 'c1out', r: pos.i, c: pos.j }],
        links: [{ a: 'win', b: 'cell', cone: true }],
      });
      const btn = button('▶ Sweep', () => {
        if (sweeping) { sweeping = false; TICK = null; sweep = 576; btn.textContent = '▶ Sweep'; kick(); return; }
        sweeping = true; sweep = 0; btn.textContent = '■ Stop';
        let last = 0;
        TICK = (now) => {
          if (now - last < 16) return true; last = now;
          sweep = Math.min(576, sweep + 3);
          const t = sweep - 1, i = (t / 24) | 0, j = t % 24;
          setHl('win', { r: i, c: j }); setHl('cell', { r: i, c: j });
          if (sweep % 6 === 0 || sweep === 576) setCard(cardConv1(ch, i, j));
          if (sweep >= 576) { sweeping = false; btn.textContent = '▶ Sweep'; TICK = null; return false; }
          return true;
        };
        kick();
      }, 'Compute all 576 outputs of this map');
    },
    leave() { sweeping = false; TICK = null; },
  };
}
function tileXY(k, size, gap = 3, cols = 4) { return { x: (k % cols) * (size + gap), y: ((k / cols) | 0) * (size + gap) }; }
function stepMaps() {
  return {
    title: '16 filters, 16 feature maps',
    body: `<p>All 16 filters slide over the same image, giving 16 maps of 24×24. Each map lights up where its pattern appears: one likes left edges, another the top of a stroke, another the inside of the loop.</p>
      <p>On the chip, each map's 25 × 576 multiply-adds run first, then a separate small task (<code>task_ds_add</code>) adds that filter's bias to all 576 cells.</p>`,
    enter() {
      setFig(null);
      const b = WT.conv1_b;
      setCard(`<h3>conv1 biases (stored integers)</h3><div class="eq" style="white-space:normal">${b.join(', ')}</div>
        <p class="muted">Most are 0: the trained biases were tiny, below 1/32.</p>
        <div class="row"><span>multiplies</span><span>16 × 576 × 25 = 230,400</span></div>`);
      const sheets = [];
      for (let k = 0; k < 16; k++) { const p = tileXY(k, 24); sheets.push(sConv1(k, 'conv1', { ...p, label: k === 0 ? 'maps 0 … 15, row by row · 24×24 each' : null, labelTop: true })); }
      scene({ cam: { ...FRONT, box: [-1, -4, 106, 106] }, sheets });
    },
  };
}
function stepRelu() {
  return {
    title: 'ReLU: negatives become 0',
    body: `<p><b>ReLU</b> keeps positive numbers and replaces every negative one with 0. Watch the blue cells disappear.</p>
      <p>It sounds too simple to matter, but without it, stacking layers would collapse into one big weighted sum. On the chip it is one loop over the buffer (<code>task_relu</code>), writing into the other buffer.</p>`,
    enter() {
      setFig(null);
      setCard(`<h3>ReLU on conv1</h3><div class="row"><span>values</span><span>9,216</span></div>
        <div class="row"><span>set to 0</span><span class="neg">${fmt(STAT[RUNKEY].neg1)}</span></div>
        <div class="eq">relu(x) = x &lt; 0 ? 0 : x</div>`);
      const sheets = [];
      for (let k = 0; k < 16; k++) { const p = tileXY(k, 24); sheets.push(sConv1(k, 'relu1', { ...p, label: k === 0 ? 'maps 0 … 15, row by row · 24×24 each' : null, labelTop: true })); }
      scene({ cam: { ...FRONT, box: [-1, -4, 106, 106] }, sheets });
    },
  };
}
function stepPool() {
  return {
    title: 'Max-pool: keep the strongest of four',
    body: `<p><b>Pooling</b> shrinks each map from 24×24 to 12×12: cut it into 2×2 blocks and keep only the largest value of each.</p>
      <p>The next layer gets 4× fewer numbers to process, and a stroke that moves by one pixel still lands in the same block. Hover a pooled cell to see its four candidates.</p>`,
    enter() {
      setFig(null);
      chanControl('map', () => DATA.ch, v => { DATA.ch = v; });
      const ch = DATA.ch, run = R();
      // start on the block with the largest pooled value
      const t = argmaxPos(run.pool1, ch * 144, 144); let a = (t / 12) | 0, b = t % 12;
      const show = (r, c) => {
        const S = run.relu1, off = ch * 576;
        const f = [S[off + 2 * r * 24 + 2 * c], S[off + 2 * r * 24 + 2 * c + 1], S[off + (2 * r + 1) * 24 + 2 * c], S[off + (2 * r + 1) * 24 + 2 * c + 1]];
        setCard(`<h3>pool · map ${ch} · (${r}, ${c})</h3><div class="eq">max(${f.join(', ')}) = <b>${run.pool1[ch * 144 + r * 12 + c]}</b></div>
          <div class="row"><span>24×24 → 12×12</span><span>9,216 → 2,304 values</span></div>`);
        setHl('pw', { r: 2 * r, c: 2 * c }); setHl('pc', { r, c });
      };
      show(a, b);
      scene({
        cam: { ...FRONT, box: [-1, -1, 55, 26] },
        sheets: [sConv1(ch, 'relu1', { x: 0, y: 0, label: `after ReLU · map ${ch}` }),
          sPool1(ch, { x: 30, y: 0, s: 2, label: `pooled · 12×12`, onHover: (r, c) => show(r, c) })],
        hls: [{ id: 'pw', sheet: `c1_${ch}`, r: 2 * a, c: 2 * b, h: 2, w: 2 }, { id: 'pc', sheet: `p1_${ch}`, r: a, c: b }],
        links: [{ a: 'pw', b: 'pc', cone: true }],
      });
    },
  };
}
function cardConv2(f, i, j) {
  const d = conv2Detail(f, i, j), m = Math.max(...d.per.map(Math.abs), 1);
  const bars = d.per.map((p, k) => `<span>map ${k}</span><span class="bar"><i style="${p >= 0 ? 'left:50%' : `right:50%`};width:${Math.abs(p) / m * 50}%;background:var(${p >= 0 ? '--pos' : '--neg'})"></i></span><span>${p}</span>`).join('');
  return `<h3>conv2 · map ${f} · row ${i}, col ${j}</h3>
    <p class="muted">25 products from each of the 16 input maps:</p>
    <div class="am-bars">${bars}</div>
    <div class="eq">Σ ${d.acc} + bias ${d.b} = <b>${d.out}</b></div>
    <div class="row"><span>real value</span><span>${real(d.out)}</span></div>`;
}
function stepConv2() {
  return {
    title: 'conv2 looks at all 16 maps at once',
    body: `<p>The second convolution again has 16 filters, but each filter is now 16 × 5 × 5 = <b>400 weights</b>: one 5×5 slice for each of the 16 pooled maps.</p>
      <p>One output number = the same 5×5 window on every input map, 400 products in all, plus a bias. The card shows how much each input map contributed. The 12×12 maps shrink to 8×8.</p>
      <p>Drag to turn the stack. Hover an output cell to move the window.</p>`,
    enter() {
      setFig(null);
      chanControl('filter', () => DATA.f, v => { DATA.f = v; });
      const f = DATA.f, run = R();
      const t = argmaxPos(run.conv2, f * 64, 64); const i = (t / 8) | 0, j = t % 8;
      setCard(cardConv2(f, i, j));
      const sheets = [], hls = [];
      for (let k = 0; k < 16; k++) {
        sheets.push(sPool1(k, { x: 0, y: 0, z: -k * 1.3, label: k === 0 ? 'pooled maps · 16 × 12×12' : null }));
        hls.push({ id: `w_${k}`, sheet: `p1_${k}`, r: i, c: j, h: 5, w: 5, fill: 0.08, lw: 1.5 });
      }
      sheets.push(sConv2(f, 'conv2', 'c2sel', { x: 27, y: 1, s: 1.3, label: `conv2 map ${f} · 8×8`,
        onHover: (r, c) => { for (let k = 0; k < 16; k++) setHl(`w_${k}`, { r, c }); setHl('c2cell', { r, c }); setCard(cardConv2(f, r, c)); } }));
      hls.push({ id: 'c2cell', sheet: 'c2sel', r: i, c: j });
      scene({ cam: { yaw: -0.6, pitch: -0.3, box: [-1, -1, 38.5, 14, -19.5, 0] }, sheets, hls,
        links: [{ a: 'w_0', b: 'c2cell', cone: true }], orbit: true });
    },
  };
}
function stepPool2() {
  return {
    title: 'ReLU and pool again',
    body: `<p>Same two steps as before. ReLU zeroes the negatives, then 2×2 max-pooling shrinks each 8×8 map to 4×4.</p>
      <p>What's left: 16 maps × 4 × 4 = <b>256 numbers</b>, a compact summary of which patterns appeared roughly where. That's all the rest of the network gets to see of the original 784 pixels.</p>`,
    enter() {
      setFig(null);
      setCard(`<h3>conv2 → ReLU → pool</h3>
        <div class="row"><span>conv2 values</span><span>1,024</span></div>
        <div class="row"><span>zeroed by ReLU</span><span class="neg">${fmt(STAT[RUNKEY].neg2)}</span></div>
        <div class="row"><span>after pooling</span><span>256</span></div>
        <div class="row"><span>multiplies in conv2</span><span>409,600</span></div>`);
      const sheets = [];
      for (let k = 0; k < 16; k++) {
        sheets.push(sConv2(k, 'conv2', `c2_${k}`, { x: 0, y: 0, z: -k * 0.8, s: 1.2, label: k === 0 ? 'conv2' : null }));
        sheets.push(sConv2(k, 'relu2', `r2_${k}`, { x: 18, y: 0, z: -k * 0.8, s: 1.2, label: k === 0 ? 'ReLU' : null }));
        sheets.push(sPool2(k, { x: 36, y: 2.4, z: -k * 0.8, s: 1.2, label: k === 0 ? 'pool · 4×4' : null }));
      }
      scene({ cam: { ...OBL, box: [-1, -1, 48, 12, -12, 0] }, sheets, orbit: true });
    },
  };
}
function stepFlatten() {
  return {
    title: 'Flatten: 256 numbers in a row',
    body: `<p>The fully connected layers want one long list. In memory the 16 maps already sit one after another, so "flattening" moves nothing: main.c just re-labels the buffer as 256 × 1 (<code>MAT_RESHAPE</code>). Map 0's 16 numbers come first, then map 1's, and so on.</p>
      <p>TensorFlow flattens in a different order (position first, then channel), so the export notebook reshuffles fc1's weights (<code>change_c_wh</code>) to match the chip's order. Hover a cell in the strip to see where it came from.</p>`,
    enter() {
      setFig(null);
      setCard(`<h3>Flatten</h3><div class="eq">flat[16·map + 4·row + col]</div><p class="muted">16 maps × 16 values = 256</p>`);
      const sheets = [];
      for (let k = 0; k < 16; k++) {
        sheets.push(sPool2(k, { x: k * 5.5, y: 0, s: 1, label: k === 0 ? 'pooled maps 0 … 15' : null }));
        sheets.push(sFlatSeg(k, { x: k * 5.5, y: 7.5, sx: 0.34375, sy: 2.2, from: { x: k * 5.5, y: 0, a: 0 }, label: k === 0 ? 'flat · 256' : null }));
      }
      scene({ cam: { ...FRONT, box: [-1, -1.5, 88.5, 11.5] }, sheets });
    },
  };
}
function cardFc1(k) {
  const run = R(), d = fcDetail('fc1_w', 32, 256, run.pool2, k, null);
  const top = d.prods.map((p, j) => [p, j]).sort((a, b) => Math.abs(b[0]) - Math.abs(a[0])).slice(0, 4);
  return `<h3>fc1 · output ${k}</h3>
    <div class="eq">Σ<sub>j</sub> f_mul(W[${k}][j], flat[j]) = <b>${d.acc}</b></div>
    <p class="muted">largest of the 256 products:</p>
    ${top.map(([p, j]) => `<div class="row"><span>W ${WT.fc1_w[k * 256 + j]} × flat[${j}] ${run.pool2[j]}</span><span class="${p >= 0 ? 'pos' : 'neg'}">${p}</span></div>`).join('')}
    <div class="row" style="margin-top:4px"><span>no bias, then ReLU</span><span>${run.relu3[k]} = ${real(run.relu3[k])}</span></div>`;
}
function stepFc1() {
  return {
    title: 'fc1: 256 → 32',
    body: `<p>Each of fc1's 32 outputs is a <b>dot product</b>: multiply the 256 inputs by one row of a 32 × 256 weight matrix and add everything up. Then ReLU.</p>
      <p>One thing differs from the trained model: there is <b>no bias</b> here. In main.c the bias pointer for this layer is <code>NULL</code> (<code>&amp;mat_fc1_b</code> is commented out), so the 32 trained fc1 biases are stored but never used on the device.</p>
      <p>Hover a weight, or an output on the right to pick its row.</p>`,
    enter() {
      setFig(null);
      const run = R(); const k0 = argmaxPos(run.relu3, 0, 32);
      setCard(cardFc1(k0));
      const sheets = [];
      for (let k = 0; k < 16; k++) sheets.push(sFlatSeg(k, { x: k * 5.5, y: 0, sx: 0.34375, sy: 2.2, label: k === 0 ? 'flat · 256 inputs' : null, labelTop: true }));
      const W1 = wV('fc1_w', 0, 256);
      sheets.push({ id: 'W1', rows: 32, cols: 256, v: W1, key: 'W1', pal: 'div', scale: WS.fc1_w, x: 0, y: 3.4, sx: 0.34375, sy: 0.9, label: 'fc1 weights · 32 × 256',
        info: (r, c) => { const w = W1(r, c), x = run.pool2[c]; return { html: `${tipHead(`W1[${r}][${c}]`)}${w} (${real(w)}) × flat ${x}<br>= ${fmul(w, x)}`, src: [{ sheet: `fl_${(c / 16) | 0}`, r: 0, c: c % 16 }] }; },
        onHover: (r) => pick(r) });
      const pick = (k) => { setHl('row1', { r: k }); setHl('o1', { r: k }); setHl('o1r', { r: k }); setCard(cardFc1(k)); };
      sheets.push(sVec('f1', 'fc1', 32, { x: 90.5, y: 3.4, sx: 2.2, sy: 0.9, label: 'Σ', info: infoFc1, onHover: pick }));
      sheets.push(sVec('r3', 'relu3', 32, { x: 94, y: 3.4, sx: 2.2, sy: 0.9, label: 'ReLU', info: infoFc1, onHover: pick }));
      scene({ cam: { ...FRONT, box: [-1, -3, 97, 34.5] }, sheets,
        hls: [{ id: 'row1', sheet: 'W1', r: k0, c: 0, h: 1, w: 256, fill: 0.15 }, { id: 'o1', sheet: 'f1', r: k0, c: 0 }, { id: 'o1r', sheet: 'r3', r: k0, c: 0 }],
        links: [{ a: 'row1', b: 'o1', dash: true }] });
    },
  };
}
function cardFc2(d) {
  const run = R(), x = run.relu3, r = fcDetail('fc2_w', 10, 32, x, d, WT.fc2_b[d]);
  return `<h3>fc2 · score for digit ${d}</h3>
    <div class="eq">Σ 32 products = ${r.acc}<br>+ bias ${WT.fc2_b[d]} = <b>${r.out}</b></div>
    <div class="row"><span>real value</span><span>${real(r.out)}</span></div>
    <div class="row"><span>answer</span><span class="acc">digit ${run.prediction}</span></div>`;
}
function stepFc2() {
  return {
    title: 'fc2: 32 → 10 scores',
    body: `<p>The last layer: a 10 × 32 weight matrix times the 32 numbers from fc1, plus one bias per digit. The 10 results are the <b>scores</b> (logits), one per digit 0–9.</p>
      <p>No ReLU at the end: a score can be negative. Hover a score to see its row.</p>`,
    enter() {
      setFig(null);
      const run = R(), d0 = run.prediction;
      setCard(cardFc2(d0));
      const x = run.relu3, W2 = wV('fc2_w', 0, 32);
      const pick = d => { setHl('row2', { r: d }); setHl('o2', { r: d }); setCard(cardFc2(d)); };
      const sheets = [
        { id: 'r3row', rows: 1, cols: 32, v: (r, c) => x[c], key: RUNKEY + ':r3row', pal: 'div', scale: sc('relu3'), x: 0, y: 0, sx: 2.75, sy: 2.2, label: 'fc1 after ReLU · 32', labelTop: true,
          info: (r, c) => ({ html: `${tipHead(`fc1[${c}] after ReLU`)}${x[c]} = ${real(x[c])}` }) },
        { id: 'W2', rows: 10, cols: 32, v: W2, key: 'W2', pal: 'div', scale: WS.fc2_w, x: 0, y: 3.5, sx: 2.75, sy: 2.2, label: 'fc2 weights · 10 × 32',
          info: (r, c) => ({ html: `${tipHead(`W2[${r}][${c}]`)}${W2(r, c)} (${real(W2(r, c))}) × ${x[c]} = ${fmul(W2(r, c), x[c])}`, src: [{ sheet: 'r3row', r: 0, c }] }), onHover: r => pick(r) },
        sVec('f2acc', 'fc2_acc', 10, { x: 90.5, y: 3.5, sx: 2.2, sy: 2.2, scale: sc('logits'), label: 'Σ', info: infoLogit, onHover: pick }),
        { id: 'f2b', rows: 10, cols: 1, v: r => WT.fc2_b[r], key: 'f2b', pal: 'div', scale: sc('logits'), x: 94, y: 3.5, sx: 2.2, sy: 2.2, label: 'bias', info: r => ({ html: `fc2 bias ${r}: ${WT.fc2_b[r]} = ${real(WT.fc2_b[r])}` }) },
        sVec('lg', 'logits', 10, { x: 97.5, y: 3.5, sx: 2.2, sy: 2.2, label: 'score', info: infoLogit, onHover: pick }),
      ];
      scene({ cam: { ...FRONT, box: [-1, -3, 100.5, 26.5] }, sheets,
        hls: [{ id: 'row2', sheet: 'W2', r: d0, c: 0, h: 1, w: 32, fill: 0.15 }, { id: 'o2', sheet: 'lg', r: d0, c: 0 }],
        links: [{ a: 'row2', b: 'o2', dash: true }] });
    },
  };
}
function stepAnswer() {
  return {
    title: 'The answer',
    body: `<p>The chip picks the digit with the highest score (<code>task_finish</code>) and prints all ten over the serial port. Here that's <b data-v="pred"></b>.</p>
      <p>Scores are not probabilities, and they don't add up to anything in particular; only which one is biggest matters.</p>
      <p>A small detail in main.c: it switches on the board's LED (pin P1.0) only when the answer is 7.</p>`,
    enter() { setCard(null); setFig(figLogits()); scene({ cam: OVERVIEW_CAM, sheets: [] }); },
  };
}
function stepMatch() {
  return {
    title: 'Does it match the chip?',
    body: `<div data-v="matchtext"></div>
      <p>For accuracy, the only number in the repo is from the notebook: the float TensorFlow model scored <b>98.83%</b> on the 10,000 MNIST test images. That was measured in Colab with ordinary decimals, not on the chip with integers.</p>`,
    enter() { setCard(null); setFig(figMatch()); scene({ cam: OVERVIEW_CAM, sheets: [] }); },
  };
}
function stepPower() {
  return {
    title: 'Power comes and goes',
    body: `<p>There's no battery. The solar panel slowly charges a storage capacitor. Once the voltage is high enough, the chip wakes up and computes, draining the capacitor faster than the panel refills it. When the voltage sinks too low, the chip dies, mid-layer, mid-loop, wherever it happens to be.</p>
      <p>Then the capacitor recharges and the chip boots again from scratch: registers empty, program counter back at the start. A normal program would have to start the whole inference over, and might never finish.</p>
      <p class="note">The curve is a schematic picture of the pattern, not measured data.</p>`,
    enter() {
      setCard(null); setFig(figPower()); scene({ cam: OVERVIEW_CAM, sheets: [] });
      const dot = FIG.querySelector('#am-vdot'), st = FIG.querySelector('#am-chipstate'), chip = FIG.querySelector('#am-chip');
      const X = t => 60 + t * 170, Y = v => 360 - v * 170; const t0 = performance.now();
      TICK = (now) => {
        if (!dot) return false;
        const t = ((now - t0) / 4000) % 3, p = vShape(t);
        dot.setAttribute('cx', X(t)); dot.setAttribute('cy', Y(p.v));
        st.textContent = p.on ? 'computing' : 'off · charging';
        chip.style.fill = p.on ? 'color-mix(in srgb, var(--good) 22%, var(--zero))' : 'var(--zero)';
        return true;
      };
      kick();
    },
    leave() { TICK = null; },
  };
}
// one-weight-at-a-time simulation shared by the last two grid steps
function bufSheets(ch, sim) {
  const P = sim.P;
  const writing = () => ((sim.p % 2 === 0) ? 'X' : 'Y');
  // how many products the running total in this buffer cell currently holds (null: nothing yet)
  const count = (buf, t) => {
    const p = sim.p;
    if (buf === writing()) return t < sim.s ? p + 1 : (p >= 2 ? p - 1 : null);
    return p >= 1 ? p : null;
  };
  const val = (buf, t) => { const n = count(buf, t); return n == null ? null : P[n][t]; };
  const role = buf => (writing() === buf ? 'writing' : 'reading');
  const inf = buf => (r, c) => {
    const t = r * 24 + c, n = count(buf, t);
    return { html: `${tipHead(`buffer ${buf === 'X' ? 'A' : 'B'} · (${r}, ${c})`)}${n == null ? 'nothing written yet' : `${P[n][t]}: running total of the first ${n} of 25 products`}` };
  };
  return [
    { id: 'bufX', rows: 24, cols: 24, v: (r, c) => val('X', r * 24 + c), key: 'bufX', pal: 'div', scale: sc('conv1'), x: 31, y: 0, label: () => `buffer A · ${role('X')}`, info: inf('X') },
    { id: 'bufY', rows: 24, cols: 24, v: (r, c) => val('Y', r * 24 + c), key: 'bufY', pal: 'div', scale: sc('conv1'), x: 31, y: 31, label: () => `buffer B · ${role('Y')}`, info: inf('Y') },
  ];
}
function simHls(sim) {
  const p = sim.p, l = (p / 5) | 0, n = p % 5, t = Math.max(0, Math.min(575, sim.s)), i = (t / 24) | 0, j = t % 24;
  const w = p % 2 === 0 ? 'bufX' : 'bufY', rd = p % 2 === 0 ? 'bufY' : 'bufX';
  return [
    { id: 'region', sheet: 'in', r: l, c: n, h: 24, w: 24, color: 'flow', fill: 0.06, lw: 1.5 },
    { id: 'kc', sheet: 'k1big', r: l, c: n },
    { id: 'px', sheet: 'in', r: i + l, c: j + n, color: 'ink', fill: 0 },
    { id: 'wc', sheet: w, r: i, c: j },
    { id: 'rc', sheet: rd, r: i, c: j, color: 'flow', a: p >= 1 ? 1 : 0 },
  ];
}
function moveSim(sim) {
  const p = sim.p, l = (p / 5) | 0, n = p % 5, t = Math.max(0, Math.min(575, sim.s)), i = (t / 24) | 0, j = t % 24;
  setHl('region', { r: l, c: n }); setHl('kc', { r: l, c: n }); setHl('px', { r: i + l, c: j + n }); setHl('wc', { r: i, c: j }); setHl('rc', { r: i, c: j });
}
function simCard(ch, sim, extra = '') {
  const p = sim.p, l = (p / 5) | 0, n = p % 5, w = WT.conv1_w[ch * 25 + p];
  const t = Math.max(0, Math.min(575, sim.s - 1)), i = (t / 24) | 0, j = t % 24;
  const wr = p % 2 === 0 ? 'A' : 'B', rd = p % 2 === 0 ? 'B' : 'A';
  const x = R().input[(i + l) * 28 + j + n], prod = fmul(w, x);
  return `<h3>filter ${ch} · pass ${p + 1} of 25</h3>
    <div class="row"><span>weight (${l}, ${n})</span><span>${w} = ${real(w)}</span></div>
    <div class="row"><span>cell</span><span>row ${i}, col ${j}</span></div>
    <div class="eq">${p === 0 ? `${wr} = f_mul(${w}, ${x}) = ${prod}` : `${wr} = ${rd} + f_mul(${w}, ${x})<br>&nbsp;&nbsp;= ${sim.P[p][t]} + ${prod} = <b>${sim.P[p + 1][t]}</b>`}</div>
    <div class="row"><span>read from</span><span>${p === 0 ? '— (first pass)' : 'buffer ' + rd}</span></div>
    <div class="row"><span>write to</span><span>buffer ${wr}</span></div>${extra}`;
}
function stepOrder() {
  let sim = null, paused = false;
  return {
    title: 'How the chip really orders the math',
    body: `<p>On a laptop you'd compute one output cell at a time: all 25 products for that cell, then move on. SONIC turns it inside out: take <b>one weight</b>, sweep it across the whole map, adding its product into every cell's running total. Then the next weight. 25 passes per filter.</p>
      <p>Each pass reads the running totals from one buffer and writes the new totals into the other; the next pass swaps them (<code>CUR_SCRATCH[3]</code> flips). It looks wasteful, but the next chapter shows why it matters.</p>
      <p>Blue outline on the input: all the pixels this weight touches during the pass.</p>`,
    enter() {
      setFig(null);
      chanControl('filter', () => DATA.ch, v => { DATA.ch = v; });
      const ch = DATA.ch;
      sim = { P: conv1Partials(ch), p: 0, s: 0 };
      setCard(simCard(ch, sim));
      scene({ cam: { ...FRONT, box: [-1, -1.5, 56, 57.5] },
        sheets: [sInput({ label: 'input' }), bigFilterSheet(ch, { x: 9.5, y: 33, s: 1.8, label: `filter ${ch}` }), ...bufSheets(ch, sim)],
        hls: simHls(sim) });
      let last = 0;
      const btn = button('❚❚ Pause', () => { paused = !paused; btn.textContent = paused ? '▶ Play' : '❚❚ Pause'; if (!paused) kick(); });
      TICK = (now) => {
        if (paused) return false;
        if (now - last < 16) return true; last = now;
        sim.s += 12;
        if (sim.s > 576) { sim.s = 0; sim.p = (sim.p + 1) % 25; }
        moveSim(sim);
        if (sim.s % 48 === 0) setCard(simCard(ch, sim));
        return true;
      };
      kick();
    },
    leave() { TICK = null; },
  };
}
function stepSurvive() {
  return {
    title: 'What survives a blackout',
    body: `<p>When power dies, everything in SRAM is gone. But SONIC keeps all the state that matters in FRAM: the weights, the feature-map buffers, which task was running, and that task's loop counters.</p>
      <p>The tricky part is updating that state safely. If power fails halfway through changing two related numbers, you could wake up with one new and one old. SONIC (through libalpaca) never overwrites progress in place: it writes the new values to a list first, then switches over with a single pointer write.</p>`,
    enter() { setCard(null); setFig(figSurvive()); scene({ cam: OVERVIEW_CAM, sheets: [] }); },
  };
}
function stepResume() {
  let sim = null;
  return {
    title: 'Resume, don\'t restart',
    body: `<p>Here's the same pass as before, with power cutting out. When the chip wakes up, <code>main()</code> jumps straight back into <code>task_dm_conv</code>, and the loop starts at the row and column saved in FRAM (<code>CUR_SCRATCH[4]</code> and <code>[5]</code>, updated after every cell).</p>
      <p>Why is re-doing the interrupted cell safe? Because a pass only <b>reads</b> one buffer and only <b>writes</b> the other. Recomputing a cell reads the same untouched old total and writes the same new one. If it added into the same buffer in place, a cell caught mid-update could be counted twice.</p>
      <p class="note">Timing and voltage here are illustrative. The arithmetic in the buffers is the real conv1 computation.</p>`,
    enter() {
      setFig(null);
      const ch = DATA.ch;
      sim = { P: conv1Partials(ch), p: 4, s: 0 };
      const st = { phase: 'run', until: 0, v: 1, cuts: 0, saved: 0, left: 300, total: 300 };
      const card = () => {
        const pct = Math.round(st.v * 100);
        setCard(simCard(ch, sim, `<div style="margin-top:8px" class="row"><span>capacitor (illustrative)</span><span>${st.phase === 'run' ? 'computing' : st.phase === 'off' ? 'off, recharging' : 'booting'}</span></div>
          <div class="am-gauge"><i style="width:${pct}%;background:var(${st.phase === 'run' ? '--good' : '--bad'})"></i><b style="left:30%"></b></div>
          <div class="row"><span>blackouts so far</span><span>${st.cuts}</span></div>
          <div class="row"><span>cells redone after them</span><span>${st.cuts}</span></div>`));
      };
      card();
      scene({ cam: { ...FRONT, box: [-1, -1.5, 56, 57.5] },
        sheets: [sInput({ label: 'input' }), bigFilterSheet(ch, { x: 9.5, y: 33, s: 1.8, label: `filter ${ch}` }), ...bufSheets(ch, sim)],
        hls: simHls(sim) });
      let last = 0, frameN = 0;
      TICK = (now) => {
        if (now - last < 16) return true; last = now; frameN++;
        if (st.phase === 'run') {
          sim.s += 6; st.left -= 6; st.v = 0.3 + 0.7 * Math.max(0, st.left) / st.total;
          if (sim.s > 576) { sim.s = 0; sim.p = (sim.p + 1) % 25; }
          if (st.left <= 0) {
            st.phase = 'off'; st.until = now + 1700; st.cuts++; st.saved = sim.s;
            DIM = 0.35; badge(`power lost at row ${(sim.s / 24) | 0}, col ${sim.s % 24}<br><span style="font-weight:400;font-size:12px">FRAM still holds both buffers and the loop position</span>`, 'off');
          }
          moveSim(sim);
        } else if (st.phase === 'off') {
          st.v = Math.min(1, st.v + 0.012);
          if (now > st.until) { st.phase = 'boot'; st.until = now + 1500; DIM = 1;
            badge(`power back · task_prologue → task_dm_conv<br><span style="font-weight:400;font-size:12px">resume at row ${(st.saved / 24) | 0}, col ${st.saved % 24}; that one cell is computed again</span>`, 'on'); }
        } else if (st.phase === 'boot') {
          if (now > st.until) { st.phase = 'run'; badge(null); sim.s = Math.max(0, st.saved - 1); st.total = st.left = 240 + ((st.cuts * 137) % 360); }
        }
        if (frameN % 6 === 0) card();
        return true;
      };
      kick();
    },
    leave() { TICK = null; DIM = 1; badge(null); },
  };
}
function stepTasks() {
  return {
    title: 'The whole run, counted',
    body: `<p>The full inference is one long chain of small tasks. <code>task_compute</code> walks through states 0 to 9, and each state launches a layer task that is itself split into passes.</p>
      <p>Because every pass is its own checkpointed task, a blackout costs at most one pass, and with the saved row and column, really just one cell. Without it, every blackout would throw away up to 648,512 multiplies and start over.</p>
      <p>That is how a 16 MHz chip with 8 KB of working memory, running on a patch of light, gets all the way to "${'<span data-v="pred"></span>'}".</p>`,
    enter() { setCard(null); setFig(figTasks()); scene({ cam: OVERVIEW_CAM, sheets: [] }); },
  };
}

// ---------------------------------------------------------------- boot
function wrap(step) {
  return {
    title: step.title, body: step.body,
    enter(ctx) {
      TICK = null; DIM = 1; badge(null); HOVER = null;
      UYAW = 0; UPITCH = 0;
      runControl();
      fillVals();
      step.enter(ctx);
    },
    leave(ctx) { if (step.leave) step.leave(ctx); TICK = null; },
  };
}

async function loadJSON(name) {
  const res = await fetch(new URL('./data/' + name, import.meta.url));
  if (!res.ok) throw new Error(name + ': ' + res.status);
  return res.json();
}

async function init(ctx) {
  CTX = ctx; STAGE = document.getElementById('stage');
  CANVAS = document.createElement('canvas'); CANVAS.className = 'am-canvas';
  CANVAS.setAttribute('aria-label', 'Network diagram: hover or tap a cell to read its value');
  FIG = document.createElement('div'); FIG.className = 'am-fig';
  CARD = document.createElement('div'); CARD.className = 'am-card'; CARD.setAttribute('aria-live', 'polite');
  BADGE = document.createElement('div'); BADGE.className = 'am-badge';
  ctx.stage.append(CANVAS, FIG, CARD, BADGE);
  G = CANVAS.getContext('2d');
  try {
    const [w, one, zero] = await Promise.all([loadJSON('weights.json'), loadJSON('run-one.json'), loadJSON('run-zero.json').catch(() => null)]);
    WT = w; RUNS.one = one; if (zero) RUNS.zero = zero; else RUNKEY = 'one';
  } catch (e) {
    ctx.stage.innerHTML = `<p style="padding:24px">Could not load the data files (${esc(e.message)}). If you opened this file directly, serve the folder with <code>python3 -m http.server</code>.</p>`;
    throw e;
  }
  for (const k of Object.keys(RUNS)) STAT[k] = prepRun(RUNS[k]);
  WS = { conv1_w: maxAbs(WT.conv1_w), conv2_w: maxAbs(WT.conv2_w), fc1_w: maxAbs(WT.fc1_w), fc2_w: maxAbs(WT.fc2_w) };
  readColors();
  const resize = () => {
    const r = STAGE.getBoundingClientRect(); SW = r.width; SH = r.height;
    DPR = Math.min(2, window.devicePixelRatio || 1);
    CANVAS.width = Math.max(1, Math.round(SW * DPR)); CANVAS.height = Math.max(1, Math.round(SH * DPR));
    if (SPEC) { CAM.tgt = camFor(SPEC.cam); if (!TWEENING) CAM.cur = { ...CAM.tgt }; }
    kick();
  };
  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(STAGE); else addEventListener('resize', resize);
  const retheme = () => { readColors(); kick(); };
  addEventListener('themechange', retheme);
  try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', retheme); } catch (e) { /* old browsers */ }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { FONT = css('--font-body') || FONT; kick(); });
  bindPointer();
}

const chapters = [
  { title: 'The chip and the sun', steps: [stepOverview(), stepMemory()] },
  { title: 'The input digit', steps: [stepInput(), stepFixed()] },
  { title: 'Conv layers', steps: [stepFilter(), stepSlide(), stepMaps(), stepRelu(), stepPool(), stepConv2(), stepPool2()] },
  { title: 'Fully connected', steps: [stepFlatten(), stepFc1(), stepFc2()] },
  { title: 'The answer', steps: [stepAnswer(), stepMatch()] },
  { title: 'Surviving power loss', steps: [stepPower(), stepOrder(), stepSurvive(), stepResume(), stepTasks()] },
].map(c => ({ title: c.title, steps: c.steps.map(wrap) }));

createStepper({
  slug: 'ambient-mnist',
  title: 'AmbientMNIST',
  credit: 'By Kevin Na · MNIST on a solar-powered MSP430 with SONIC',
  repo: 'https://github.com/kevinyjn1/AmbientMNIST',
  chapters,
  init,
});
