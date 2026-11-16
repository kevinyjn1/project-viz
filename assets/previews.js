// Small decorative animations for the project cards on the home page.
// They only hint at what each project does; the real numbers live on the project pages.
//
//   mountPreview(canvas, slug)  draws the animation for that project into the canvas.
//
// To give a new project its own preview, add a function to DRAW below with the same key
// as the project's slug. Projects without one get a simple generic animation.
import { css } from './stepper.js';

// ---------- color helpers ----------
const hex = h => {
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const ease = x => x * x * (3 - 2 * x);

function palette() {
  const get = name => { const v = css(name); return /^#[0-9a-f]{3,6}$/i.test(v) ? hex(v) : [128, 128, 128]; };
  const P = {};
  for (const k of ['ink', 'muted', 'line', 'accent', 'flow', 'neg', 'pos', 'zero', 'empty', 'cell-line', 'frame', 'stage'])
    P[k.replace('-', '')] = get('--' + k);
  return P;
}
// -1 (blue) .. 0 (neutral) .. +1 (orange)
const diverge = (P, v) => (v < 0 ? mix(P.zero, P.neg, clamp(-v)) : mix(P.zero, P.pos, clamp(v)));

// ---------- HydroSense: a tap's sound wave turns into a grid of MFCC-like cells ----------
const TAPS = [[0.06, 1], [0.37, 0.75], [0.67, 0.9]];
function wave(u) {
  let s = 0, e = 0;
  for (const [u0, a] of TAPS) {
    const d = u - u0;
    if (d < 0) continue;
    const env = a * Math.exp(-d * 13);
    s += env * Math.sin(d * 2 * Math.PI * 30);
    e += env;
  }
  s += 0.035 * Math.sin(u * 517) * Math.sin(u * 93);
  return [s, Math.min(1, e)];
}
function hydrosense(g, W, H, P, t) {
  const T = 7, ph = (t % T) / T;
  const p = clamp(ph / 0.78);                       // sweep progress
  const fade = ph > 0.93 ? 1 - (ph - 0.93) / 0.07 : 1;
  const pad = Math.max(14, W * 0.05);
  const wx0 = pad, wx1 = W * 0.47, mid = H / 2, amp = H * 0.3;
  const gx0 = W * 0.56, gx1 = W - pad;
  const C = 16, R = 10;
  const cw = (gx1 - gx0) / C, ch = Math.min((H - 2 * pad) / R, cw * 1.25);
  const gy0 = mid - (ch * R) / 2;

  // baseline + wave
  g.strokeStyle = rgb(P.line); g.lineWidth = 1;
  g.beginPath(); g.moveTo(wx0, mid); g.lineTo(wx1, mid); g.stroke();
  g.strokeStyle = rgb(P.flow); g.lineWidth = 1.4; g.beginPath();
  const N = Math.max(80, Math.round(wx1 - wx0));
  for (let i = 0; i <= N; i++) {
    const u = i / N, x = wx0 + u * (wx1 - wx0), y = mid - wave(u)[0] * amp;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();

  // sliding window over the wave
  const ww = (wx1 - wx0) / C * 1.6;
  const wxc = wx0 + p * (wx1 - wx0);
  const wl = clamp(wxc - ww / 2, wx0, wx1 - ww);
  g.fillStyle = rgb(P.accent, 0.12 * fade); g.strokeStyle = rgb(P.accent, fade); g.lineWidth = 1.5;
  g.fillRect(wl, mid - amp - 8, ww, 2 * amp + 16);
  g.strokeRect(wl + 0.5, mid - amp - 8 + 0.5, ww - 1, 2 * amp + 15);

  // grid of cells, filled column by column as the window passes
  const done = p * C;
  for (let j = 0; j < C; j++) {
    const [, e] = wave((j + 0.5) / C);
    const on = clamp(done - j);
    for (let r = 0; r < R; r++) {
      const v = (0.25 + e) * Math.cos(r * 0.95 + j * 0.42 + e * 2) * (r === 0 ? 1.2 : 1 - r * 0.05);
      const c = on > 0 ? mix(P.empty, diverge(P, v), on * fade) : P.empty;
      g.fillStyle = rgb(c);
      g.fillRect(gx0 + j * cw + 0.75, gy0 + r * ch + 0.75, cw - 1.5, ch - 1.5);
    }
  }
  // a thread from the window to the column being filled
  if (p < 1) {
    const cx = gx0 + (Math.min(C - 1, Math.floor(done)) + 0.5) * cw;
    g.strokeStyle = rgb(P.accent, 0.55); g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(wl + ww, mid - amp - 8);
    g.bezierCurveTo(wl + ww + 30, gy0 - 18, cx - 10, gy0 - 18, cx, gy0 - 2); g.stroke();
    g.setLineDash([]);
  }
}
hydrosense.still = 4.3;

// ---------- Checkpoint Champion: a point riding a sigmoid curve ----------
const sig = z => 1 / (1 + Math.exp(-z));
// fixed scatter of past games: x position and which side won (deterministic, decorative)
const GAMES = Array.from({ length: 34 }, (_, i) => {
  const x = -5.6 + (i * 11.2) / 33 + Math.sin(i * 12.9898) * 0.35;
  const won = sig(x * 1.1) + Math.sin(i * 78.233) * 0.32 > 0.5;
  return [x, won, Math.sin(i * 4.1) * 0.5 + 0.5];
});
function lol(g, W, H, P, t) {
  const pad = Math.max(16, W * 0.06);
  const x0 = pad, x1 = W - pad - 22, y0 = pad, y1 = H - pad;
  const X = z => x0 + ((z + 6) / 12) * (x1 - x0);
  const Y = p => y1 - p * (y1 - y0);
  // axes and the 50% line
  g.strokeStyle = rgb(P.line); g.lineWidth = 1;
  g.beginPath(); g.moveTo(x0, y1 + 0.5); g.lineTo(x1, y1 + 0.5); g.moveTo(x0 + 0.5, y0); g.lineTo(x0 + 0.5, y1); g.stroke();
  g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x0, Y(0.5)); g.lineTo(x1, Y(0.5)); g.stroke(); g.setLineDash([]);
  // past games as dots: top row for one side winning, bottom row for the other
  for (const [x, won, j] of GAMES) {
    g.fillStyle = rgb(won ? P.pos : P.neg, 0.55);
    g.beginPath(); g.arc(X(x), won ? Y(1) + 4 + j * 6 : Y(0) - 4 - j * 6, 2.6, 0, 7); g.fill();
  }
  // the curve
  g.strokeStyle = rgb(P.ink, 0.85); g.lineWidth = 2; g.beginPath();
  for (let i = 0; i <= 120; i++) {
    const z = -6 + (i / 120) * 12;
    i ? g.lineTo(X(z), Y(sig(z))) : g.moveTo(X(z), Y(sig(z)));
  }
  g.stroke();
  // the moving game state
  const z = 4.6 * Math.sin(t * 0.7), p = sig(z), px = X(z), py = Y(p);
  g.strokeStyle = rgb(P.muted, 0.7); g.lineWidth = 1; g.setLineDash([2, 3]);
  g.beginPath(); g.moveTo(px, y1); g.lineTo(px, py); g.lineTo(x1 + 8, py); g.stroke(); g.setLineDash([]);
  const c = diverge(P, (p - 0.5) * 2);
  g.fillStyle = rgb(c); g.strokeStyle = rgb(P.ink); g.lineWidth = 1.5;
  g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill(); g.stroke();
  // a side gauge split between the two teams
  const bx = W - pad - 10, bh = y1 - y0;
  g.fillStyle = rgb(P.pos, 0.85); g.fillRect(bx, y0 + (1 - p) * bh, 10, p * bh);
  g.fillStyle = rgb(P.neg, 0.85); g.fillRect(bx, y0, 10, (1 - p) * bh);
  g.strokeStyle = rgb(P.line); g.strokeRect(bx + 0.5, y0 + 0.5, 9, bh - 1);
}
lol.still = 1.2;

// ---------- AmbientMNIST: a 3x3 kernel sliding over a 28x28 digit ----------
const DIGIT = (() => {
  const pts = [];
  for (let a = -160; a <= 90; a += 4) { const r = (a * Math.PI) / 180; pts.push([13.5 + 4.8 * Math.cos(r), 9.4 + 4.8 * Math.sin(r)]); }
  for (let a = -90; a <= 160; a += 4) { const r = (a * Math.PI) / 180; pts.push([13.5 + 5.3 * Math.cos(r), 18.7 + 5.3 * Math.sin(r)]); }
  const img = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    let d = 1e9;
    for (const [px, py] of pts) d = Math.min(d, Math.hypot(x + 0.5 - px, y + 0.5 - py));
    img[y * 28 + x] = clamp((2.1 - d) / 1.3);
  }
  // an edge-finding kernel, like a first conv layer often learns
  const K = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const out = new Float32Array(26 * 26); let m = 1e-6;
  for (let y = 0; y < 26; y++) for (let x = 0; x < 26; x++) {
    let s = 0;
    for (let k = 0; k < 9; k++) s += K[k] * img[(y + ((k / 3) | 0)) * 28 + x + (k % 3)];
    out[y * 26 + x] = s; m = Math.max(m, Math.abs(s));
  }
  for (let i = 0; i < out.length; i++) out[i] /= m;
  return { img, out };
})();
function mnist(g, W, H, P, t) {
  const T = 9, ph = (t % T) / T;
  const p = ease(clamp(ph / 0.82));
  const fade = ph > 0.94 ? 1 - (ph - 0.94) / 0.06 : 1;
  const s = Math.min((H - 28) / 28, (W - 40) / (28 + 26 * 0.82 + 3));
  const s2 = s * 0.82, gap = s * 3;
  const total = 28 * s + gap + 26 * s2;
  const ax = (W - total) / 2, ay = (H - 28 * s) / 2;
  const bx = ax + 28 * s + gap, by = ay + s;
  // input digit
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    g.fillStyle = rgb(mix(P.empty, P.ink, DIGIT.img[y * 28 + x] * 0.92));
    g.fillRect(ax + x * s, ay + y * s, s - 0.6, s - 0.6);
  }
  // output map, filled in reading order up to the kernel
  const idx = Math.min(675, Math.floor(p * 676));
  for (let i = 0; i < 676; i++) {
    const on = i <= idx && fade > 0;
    g.fillStyle = rgb(on ? mix(P.empty, diverge(P, DIGIT.out[i] * 1.3), fade) : P.empty);
    g.fillRect(bx + (i % 26) * s2, by + ((i / 26) | 0) * s2, s2 - 0.5, s2 - 0.5);
  }
  // kernel box and its link to the output cell
  const kx = idx % 26, ky = (idx / 26) | 0;
  const rx = ax + kx * s, ry = ay + ky * s;
  const ox = bx + kx * s2, oy = by + ky * s2;
  g.strokeStyle = rgb(P.accent, 0.45 * fade); g.lineWidth = 1;
  g.beginPath(); g.moveTo(rx + 3 * s, ry); g.lineTo(ox, oy); g.moveTo(rx + 3 * s, ry + 3 * s); g.lineTo(ox, oy + s2); g.stroke();
  g.fillStyle = rgb(P.accent, 0.14 * fade); g.fillRect(rx, ry, 3 * s, 3 * s);
  g.strokeStyle = rgb(P.accent, fade); g.lineWidth = 2;
  g.strokeRect(rx, ry, 3 * s, 3 * s);
  g.strokeRect(ox, oy, s2, s2);
}
mnist.still = 4.2;

// ---------- anything else: a gentle grid of cells ----------
function generic(g, W, H, P, t) {
  const n = 12, m = 7, s = Math.min((W - 40) / n, (H - 40) / m);
  const x0 = (W - n * s) / 2, y0 = (H - m * s) / 2;
  for (let y = 0; y < m; y++) for (let x = 0; x < n; x++) {
    g.fillStyle = rgb(diverge(P, Math.sin(x * 0.7 + y * 0.9 + t * 1.2) * Math.cos(y * 0.5 - t * 0.6)));
    g.fillRect(x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2);
  }
}
generic.still = 1;

const DRAW = { hydrosense, 'lol-predictor': lol, 'ambient-mnist': mnist };

export function mountPreview(canvas, slug) {
  const draw = DRAW[slug] || generic;
  const g = canvas.getContext('2d');
  if (!g) return;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  let P = palette(), W = 0, H = 0, raf = 0, visible = true;
  const t0 = performance.now();

  function size() {
    const r = canvas.getBoundingClientRect(), d = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height;
    canvas.width = Math.max(1, Math.round(W * d)); canvas.height = Math.max(1, Math.round(H * d));
    g.setTransform(d, 0, 0, d, 0, 0);
  }
  function frame(now) {
    raf = 0;
    if (W < 2 || H < 2) return;
    g.clearRect(0, 0, W, H);
    draw(g, W, H, P, reduce.matches ? draw.still : (now - t0) / 1000);
    if (visible && !reduce.matches) raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };
  const repaint = () => { P = palette(); kick(); };

  size();
  if ('ResizeObserver' in window) new ResizeObserver(() => { size(); kick(); }).observe(canvas);
  else addEventListener('resize', () => { size(); kick(); });
  if ('IntersectionObserver' in window)
    new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; if (visible) kick(); }).observe(canvas);
  addEventListener('themechange', repaint);
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', repaint);
  reduce.addEventListener?.('change', kick);
  kick();
}
