// HydroSense, under the hood: the stage drawings and the step list.
// Every number comes from data/*.json, made by tools/hydrosense/precompute.py
// from the repo's own WAV files and saved models (models/*.pkl).
import { createStepper, css } from '../assets/stepper.js';

const NS = 'http://www.w3.org/2000/svg';
const DATA = new URL('data/', import.meta.url);
let MODEL = null;          // data/model.json
const RECS = {};           // id -> recording json
let R = null;              // the recording being shown
let CTX = null, svg = null, gEls = null, gOver = null, stageEl = null;
let W = 800, H = 600, wide = true, curScene = null, lastOn = new Set();
const els = {};            // name -> element
const HOV = {};            // hover handlers by data-h key
let timers = [];

// ------------------------------------------------------------------ small helpers
function h(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, cls = 't', anchor = 'start') {
  const e = h('text', { x, y, class: cls, 'text-anchor': anchor }, parent);
  e.textContent = s;
  return e;
}
const fx = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const fint = v => Math.round(v).toLocaleString('en-US');
const hex2 = v => v.toString(16).toUpperCase().padStart(2, '0');
const clsName = c => (c === 'empty' ? 'empty' : c.replace('ml', ' ml'));
const mlOf = c => (c === 'empty' ? 0 : parseInt(c, 10));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const later = (fn, ms) => { const t = setTimeout(fn, ms); timers.push(t); return t; };
function clearTimers() { timers.forEach(t => { clearTimeout(t); clearInterval(t); }); timers = []; }

// firmware constants (Core/Src/main.c): LED_ShowBar(count, r, g, b) per class, and the command bytes
const LED = {
  empty: [2, [255, 0, 0], 'red'], '100ml': [4, [255, 0, 0], 'red'],
  '300ml': [6, [255, 255, 0], 'yellow'], '500ml': [8, [255, 255, 0], 'yellow'],
  '700ml': [10, [0, 255, 0], 'green'], '900ml': [15, [0, 255, 0], 'green'],
};
const CMD = { empty: 0x20, '100ml': 0x21, '300ml': 0x22, '500ml': 0x23, '700ml': 0x24, '900ml': 0x25 };
const SR = 44100, HOP = 512, NFFT = 2048;

// ------------------------------------------------------------------ colors (read from CSS tokens)
const P = {};
function hex2rgb(c) {
  c = (c || '').trim();
  if (c.startsWith('#')) {
    if (c.length === 4) c = '#' + [...c.slice(1)].map(x => x + x).join('');
    const n = parseInt(c.slice(1, 7), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = c.match(/[\d.]+/g);
  return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}
function readPalette() {
  for (const k of ['neg', 'pos', 'zero', 'flow', 'ink', 'muted', 'line', 'stage', 'surface', 'accent', 'empty', 'frame'])
    P[k] = hex2rgb(css('--' + k));
}
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = a => `rgb(${a[0]},${a[1]},${a[2]})`;
function divA(v) { v = clamp(v, -1, 1); return v < 0 ? mix(P.zero, P.neg, -v) : mix(P.zero, P.pos, v); }
const div = v => rgb(divA(v));
function seqA(t) { return mix(P.zero, P.pos, clamp(t, 0, 1)); }
const seq = t => rgb(seqA(t));
const clsColor = ci => rgb(mix(P.surface, P.flow, 0.22 + 0.78 * ci / 5));

// draw a grid of values into a data URL (one pixel per cell; shown with pixelated scaling)
function heatURL(rows, colorA, flipY) {
  const nr = rows.length, nc = rows[0].length;
  const cv = document.createElement('canvas');
  cv.width = nc; cv.height = nr;
  const g = cv.getContext('2d');
  const img = g.createImageData(nc, nr);
  for (let r = 0; r < nr; r++) {
    const rr = flipY ? nr - 1 - r : r;
    for (let c = 0; c < nc; c++) {
      const col = colorA(rows[r][c], r, c), o = (rr * nc + c) * 4;
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return cv.toDataURL();
}
// pointer position as a 0..1 fraction inside an svg node's on-screen box
function frac(node, e) {
  const b = node.getBoundingClientRect();
  return [clamp((e.clientX - b.left) / b.width, 0, 0.9999), clamp((e.clientY - b.top) / b.height, 0, 0.9999)];
}

// ------------------------------------------------------------------ recording-derived numbers
function derive(r) {
  const LE = MODEL.leClasses;
  r.dur = r.nSamples / r.fileSr;
  r.frameT = r.frameIdx * HOP / SR;
  r.uartFrames = Math.ceil(r.nSamples / 1024);
  r.rowMax = r.mfcc.map(row => Math.max(...row.map(Math.abs)) || 1);
  r.melLo = r.melMax - 80;
  const pi = LE.indexOf(r.pred);
  // runner-up = best other class by the model's per-class score
  let ru = -1;
  r.ovr.forEach((v, i) => { if (i !== pi && (ru < 0 || v > r.ovr[ru])) ru = i; });
  const a = Math.min(pi, ru), b = Math.max(pi, ru);
  const k = MODEL.pairs.findIndex(p => p[0] === a && p[1] === b);
  const w = MODEL.coef[k];
  const prod = r.scaled.map((x, i) => x * w[i]);
  const sum = prod.reduce((s, v) => s + v, 0);
  r.duel = { a: LE[a], b: LE[b], k, w, prod, sum, bias: MODEL.intercept[k], val: r.ovo[k] };
  r.duel.winner = r.duel.val > 0 ? r.duel.a : r.duel.b;
  r.topVotes = Math.max(...r.votes);
  r.predP = r.proba[pi];
  return r;
}

// values that step texts can show: <b data-v="name"></b>
function vars() {
  const f = MODEL.filters;
  const band = k => `${fint(f[k][0])}–${fint(f[k][2])} Hz`;
  const z0 = R.scaled[0];
  return {
    file: R.file.replace('clean_data/', ''), cls: clsName(R.cls), pred: clsName(R.pred),
    n: fint(R.nSamples), dur: R.dur.toFixed(2), dc: fint(R.dc), uartFrames: R.uartFrames,
    nFrames: R.nFrames, frameIdx: '#' + R.frameIdx, frameT: R.frameT.toFixed(2) + ' s',
    band8: band(8), band120: band(120), w8: fint(f[8][2] - f[8][0]), w120: fint(f[120][2] - f[120][0]),
    melLo: fx(R.melLo, 0), c0: fx(R.mfccCol[0], 1), c1: fx(R.mfccCol[1], 1),
    x0: fx(R.feat[0], 1), mu0: fx(MODEL.mean[0], 1), sd0: fx(MODEL.scale[0], 1), z0: fx(z0, 2),
    duelA: clsName(R.duel.a), duelB: clsName(R.duel.b), duelSum: fx(R.duel.sum, 2), duelB0: fx(R.duel.bias, 2),
    duelVal: fx(R.duel.sum + R.duel.bias, 2), duelWin: clsName(R.duel.winner), votes: R.topVotes,
    predP: (R.predP * 100).toFixed(1) + '%', byte: '0x' + hex2(CMD[R.pred]),
    ledN: LED[R.pred][0], ledC: LED[R.pred][2],
    heldNote: R.cls === R.pred ? 'correct' : `wrong: it was really ${clsName(R.cls)}`,
  };
}
function fillVars() {
  if (!R) return;
  const v = vars();
  document.querySelectorAll('#step-body [data-v]').forEach(e => { e.textContent = v[e.dataset.v] ?? ''; e.classList.add('v'); });
}

// ------------------------------------------------------------------ element framework
// An element is a <g> drawn at a fixed natural size (nw x nh); steps move and scale it.
function mk(name, nw, nh, spec) {
  const g = h('g', { class: 'el', 'data-el': name }, gEls);
  const el = { name, g, nw, nh, ...spec, box: null };
  els[name] = el;
  el.build?.(g, el);
  return el;
}
function setMode(el, m) {
  if ((el.g.dataset.mode || '') === m) return;
  el.g.dataset.mode = m;
  el.onMode?.(el, m);
}
function placeEl(el, p) {
  const sx = p.sx ?? p.s, sy = p.sy ?? p.s;
  el.g.style.transform = `translate(${p.x}px,${p.y}px) scale(${sx},${sy})`;
  el.box = { x: p.x, y: p.y, w: el.nw * sx, h: el.nh * sy };
}
function updateAll() { for (const k in els) els[k].update?.(els[k]); }

// ------------------------------------------------------------------ layout helpers
function content() {
  const pad = wide ? 18 : 10;
  return { x: pad, y: pad, w: W - 2 * pad, h: H - pad - 50 };
}
function split(r, ws, dir, gap = 18) {
  const tot = ws.reduce((a, b) => a + b, 0), out = [];
  const L = (dir === 'h' ? r.w : r.h) - gap * (ws.length - 1);
  let p = dir === 'h' ? r.x : r.y;
  for (const w of ws) {
    const len = L * w / tot;
    out.push(dir === 'h' ? { x: p, y: r.y, w: len, h: r.h } : { x: r.x, y: p, w: r.w, h: len });
    p += len + gap;
  }
  return out;
}
function fit(name, r, maxS = 1.35) {
  const e = els[name];
  const s = Math.min(r.w / e.nw, r.h / e.nh, maxS);
  return { x: r.x + (r.w - e.nw * s) / 2, y: r.y + (r.h - e.nh * s) / 2, s };
}
const auto = (ws) => split(content(), ws, wide ? 'h' : 'v');

// ------------------------------------------------------------------ scene application
function applyScene(sceneFn, animate = true) {
  curScene = sceneFn;
  const sc = sceneFn();
  if (!animate) svg.classList.add('noanim');
  const now = new Set(Object.keys(sc.p));
  for (const name in els) {
    const el = els[name];
    if (now.has(name)) {
      const fresh = !lastOn.has(name);
      if (fresh) el.g.classList.add('jump');
      placeEl(el, sc.p[name]);
      setMode(el, (sc.modes && sc.modes[name]) || '');
      el.g.classList.add('on');
      if (fresh) { el.g.getBoundingClientRect(); requestAnimationFrame(() => el.g.classList.remove('jump')); }
    } else {
      el.g.classList.remove('on');
    }
  }
  lastOn = now;
  drawOverlay(sc, animate);
  if (!animate) { svg.getBoundingClientRect(); requestAnimationFrame(() => svg.classList.remove('noanim')); }
}

function drawOverlay(sc, animate) {
  [...gOver.children].forEach(c => {
    if (!animate) { c.remove(); return; }
    c.classList.add('out'); setTimeout(() => c.remove(), 320);
  });
  const g = h('g', { class: 'ov' }, gOver);
  if (!animate) g.style.animationDelay = '0s';
  for (const a of sc.arrows || []) arrow(g, els[a[0]].box, els[a[1]].box, a[2], a[3] || 0);
  if (sc.lit) sc.lit.forEach(n => {
    const b = els[n].box;
    h('rect', { x: b.x - 5, y: b.y - 5, width: b.w + 10, height: b.h + 10, rx: 8, class: 'litbox', 'data-lit': n }, g);
  });
}
function arrow(g, A, B, label, off) {
  const ac = { x: A.x + A.w / 2, y: A.y + A.h / 2 }, bc = { x: B.x + B.w / 2, y: B.y + B.h / 2 };
  const dx = bc.x - ac.x, dy = bc.y - ac.y;
  let x1, y1, x2, y2, d;
  if (Math.abs(dx) / (A.w + B.w) >= Math.abs(dy) / (A.h + B.h)) {
    x1 = dx > 0 ? A.x + A.w + 4 : A.x - 4; x2 = dx > 0 ? B.x - 8 : B.x + B.w + 8;
    const yA = clamp(bc.y, A.y + 10, A.y + A.h - 10), yB = clamp(ac.y, B.y + 10, B.y + B.h - 10);
    y1 = (yA + yB) / 2 + off; y2 = y1;
    if (Math.abs(yA - yB) > Math.min(A.h, B.h) / 2) { y1 = ac.y + off; y2 = bc.y + off; }
    const mx = (x1 + x2) / 2;
    d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    head(g, x2, y2, dx > 0 ? 0 : 180);
  } else {
    y1 = dy > 0 ? A.y + A.h + 4 : A.y - 4; y2 = dy > 0 ? B.y - 8 : B.y + B.h + 8;
    const xA = clamp(bc.x, A.x + 10, A.x + A.w - 10), xB = clamp(ac.x, B.x + 10, B.x + B.w - 10);
    x1 = (xA + xB) / 2 + off; x2 = x1;
    if (Math.abs(xA - xB) > Math.min(A.w, B.w) / 2) { x1 = ac.x + off; x2 = bc.x + off; }
    const my = (y1 + y2) / 2;
    d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
    head(g, x2, y2, dy > 0 ? 90 : -90);
  }
  h('path', { d, class: 'arrow' }, g);
  if (label) {
    const t = txt(g, (x1 + x2) / 2, (y1 + y2) / 2 - 6, label, 'tm halo', 'middle');
    t.style.fill = 'var(--flow)';
  }
}
function head(g, x, y, ang) {
  h('path', { d: 'M0,-5 L8,0 L0,5 z', class: 'arrowhead', transform: `translate(${x},${y}) rotate(${ang})` }, g);
}

// ------------------------------------------------------------------ hover
let hlNode = null;
function onHover(e) {
  const t = e.target.closest ? e.target.closest('[data-h]') : null;
  if (hlNode && hlNode !== t) { hlNode.classList.remove('hl'); hlNode = null; }
  if (!t || !t.closest('.el.on')) { CTX.tip.hide(); return; }
  const fn = HOV[t.dataset.h];
  const html = fn ? fn(t, e) : null;
  if (!html) { CTX.tip.hide(); return; }
  if (t.tagName === 'rect' && !t.classList.contains('hit')) { t.classList.add('hl'); hlNode = t; }
  const r = stageEl.getBoundingClientRect();
  CTX.tip.show(e.clientX - r.left, e.clientY - r.top, html);
}
HOV.tip = t => t.dataset.tip;

// ------------------------------------------------------------------ the elements
function defineElements() {
  // ---- the bottle, its sensors, and the 15-LED strip
  mk('bottle', 240, 336, {
    build(g, el) {
      const BP = 'M66 36 h48 v18 q32 12 32 44 v196 q0 20 -20 20 h-72 q-20 0 -20 -20 v-196 q0 -32 32 -44 z';
      const cp = h('clipPath', { id: 'hs-bclip' }, h('defs', {}, g));
      h('path', { d: BP }, cp);
      h('rect', { x: 60, y: 16, width: 60, height: 20, rx: 4, class: 'cap' }, g);
      const wg = h('g', { 'clip-path': 'url(#hs-bclip)' }, g);
      el.water = h('rect', { x: 20, y: 314, width: 140, height: 300, class: 'water' }, wg);
      h('path', { d: BP, class: 'bottle' }, g);
      h('circle', { cx: 150, cy: 118, r: 10, class: 'ripple' }, g);
      [['VPU mic', 190], ['IMU', 226]].forEach(([s, y]) => {
        h('rect', { x: 2, y, width: 52, height: 22, rx: 4, class: 'sensor', 'data-h': 'tip',
          'data-tip': s === 'IMU' ? 'LSM6DSO32 motion sensor, set to detect a single tap (interrupt on EXTI0)' : 'VPU (voice pickup) microphone: feels vibration, outputs PDM bits' }, g);
        txt(g, 28, y + 15, s, 'tm', 'middle');
      });
      el.leds = [];
      for (let i = 0; i < 15; i++)
        el.leds.push(h('circle', { cx: 212, cy: 306 - i * 18.5, r: 7, class: 'led', 'data-h': 'tip', 'data-tip': `WS2812B LED ${i}` }, g));
      txt(g, 212, 22, 'LEDs', 'tm', 'middle');
      el.lvl = txt(g, 90, 334, '', 'ts', 'middle');
    },
    update(el) {
      const ml = mlOf(R.cls);
      el.water.style.transform = `translateY(${-ml / 1000 * 252}px)`;
      el.lvl.textContent = `really ${clsName(R.cls)}`;
      setLeds(el);
    },
    onMode(el) { setLeds(el); },
  });
  function setLeds(el) {
    if (!R) return;
    const m = el.g.dataset.mode || '';
    const [n, c] = LED[R.pred];
    el.leds.forEach((L, i) => {
      let col = null;
      if (m.includes('rec')) col = [255, 80, 0];
      else if (m.includes('result')) col = i < n ? c : null;
      else if (m.includes('idle')) col = [0, 0, 255];
      L.style.fill = col ? rgb(col) : 'var(--empty)';
    });
  }

  // ---- the two computers
  const boxEl = (name, title, sub, items) => mk(name, 232, 172, {
    build(g) {
      h('rect', { x: 1, y: 1, width: 230, height: 170, rx: 10, class: 'box' }, g);
      txt(g, 14, 24, title, 'tt');
      txt(g, 14, 41, sub, 'tm');
      items.forEach(([s, tip], i) => {
        h('rect', { x: 14, y: 54 + i * 28, width: 204, height: 22, rx: 5, class: 'pill', 'data-h': 'tip', 'data-tip': tip }, g);
        txt(g, 24, 69 + i * 28, s, 'ts').style.pointerEvents = 'none';
      });
    },
  });
  boxEl('chip', 'STM32L4R5 board', 'NUCLEO-L4R5ZI-P, main.c', [
    ['EXTI0: tap interrupt', 'The IMU pulls pin 0; HAL_GPIO_EXTI_Callback sets g_single_tap_flag'],
    ['DFSDM + DMA: mic → samples', 'Digital filter turns the 1-bit PDM stream into 44,117 samples/s'],
    ['USART3: 921,600 baud', 'Streams PCM frames to the laptop and receives the 1-byte LED command'],
    ['TIM2 PWM + DMA → LEDs', 'Generates the WS2812B bit timing for 15 LEDs'],
  ]);
  boxEl('laptop', 'Laptop', 'Python · tap_classifier.py', [
    ['save the WAV file', 'Collects frames between the START and STOP markers'],
    ['13 MFCCs → 52 numbers', 'extract_features(): librosa MFCC, then mean/std/max/min'],
    ['StandardScaler + linear SVM', 'models/scaler.pkl and models/svm_model.pkl'],
    ['send 1 byte back', 'CLASS_TO_CMD: 0x20 (empty) … 0x25 (900 ml)'],
  ]);

  // ---- tap → wait → record timeline
  mk('timeline', 590, 176, {
    build(g) {
      const X = t => 40 + t * 120;
      txt(g, 0, 16, 'What happens after a tap (times from main.c)', 'tt');
      h('rect', { x: X(0), y: 66, width: X(1) - X(0), height: 40, class: 'band-wait', 'data-h': 'tip', 'data-tip': 'PRE_RECORD_DELAY_MS = 1000' }, g);
      txt(g, X(0.5), 90, 'wait 1000 ms', 'ts', 'middle');
      h('rect', { x: X(1), y: 66, width: X(4) - X(1), height: 40, class: 'band-rec', 'data-h': 'tip', 'data-tip': 'RECORD_DURATION_SEC = 3 → SAMPLES_TO_RECORD = 44,117 × 3' }, g);
      txt(g, X(2.5), 84, 'record 3 s while you shake', 'ts', 'middle');
      txt(g, X(2.5), 99, 'LEDs blink orange (every 100 ms)', 'tm', 'middle');
      h('line', { x1: X(0), y1: 120, x2: X(4.5), y2: 120, class: 'axis' }, g);
      for (let t = 0; t <= 4; t++) { h('line', { x1: X(t), y1: 120, x2: X(t), y2: 125, class: 'axis' }, g); txt(g, X(t), 138, t + ' s', 'tm', 'middle'); }
      h('circle', { cx: X(0), cy: 120, r: 6, fill: 'var(--accent)' }, g);
      txt(g, X(0), 56, 'tap', 'ts tb', 'middle');
      txt(g, X(0) + 4, 44, 'IMU → EXTI0', 'tm');
      txt(g, X(1), 56, 'START  BB 66', 'tm', 'middle');
      txt(g, X(4), 56, 'STOP  CC 77', 'tm', 'middle');
      txt(g, X(4.25), 90, 'classify', 'tm', 'middle');
      txt(g, X(0), 162, 'Taps during recording or the 100 ms cooldown after it are ignored.', 'tm');
      h('line', { x1: X(0), y1: 60, x2: X(0), y2: 124, class: 'playhead' }, g);
    },
  });

  // ---- PDM mic → DFSDM chain
  mk('pdm', 624, 186, {
    build(g) {
      // a sketch of a 1-bit stream (first-order sigma-delta of a sine): NOT recorded data
      let acc = 0;
      for (let i = 0; i < 64; i++) {
        const v = 0.8 * Math.sin(i / 64 * 2 * Math.PI * 1.5); acc += v;
        const bit = acc > 0 ? 1 : 0; acc -= bit ? 1 : -1;
        if (bit) h('rect', { x: 2 + i * 2, y: 14, width: 1.6, height: 20, fill: 'var(--flow)' }, g);
      }
      h('line', { x1: 2, y1: 34, x2: 130, y2: 34, class: 'axis' }, g);
      txt(g, 2, 52, '1-bit stream (sketch, not data)', 'tm');
      const boxes = [
        ['PDM mic', '1 bit per clock tick', '48 MHz ÷ 17 = 2.82 MHz'],
        ['DFSDM filter', 'sinc³, oversampling 64', '64 bits → 1 number'],
        ['44,117 samples/s', '2.82 MHz ÷ 64', 'DMA: 2 × 1024 buffer'],
        ['16-bit PCM', '>>7 in hardware', '>>5 in code, clipped'],
      ];
      boxes.forEach((b, i) => {
        const x = i * 160;
        h('rect', { x, y: 70, width: 144, height: 78, rx: 8, class: 'box' }, g);
        txt(g, x + 10, 92, b[0], 'ts tb');
        txt(g, x + 10, 112, b[1], 'ts');
        txt(g, x + 10, 130, b[2], 'ts');
        if (i < 3) { h('line', { x1: x + 146, y1: 109, x2: x + 152, y2: 109, class: 'axis' }, g); head(g, x + 151, 109, 0); }
      });
      txt(g, 0, 172, 'SAMPLE_RATE = 48 MHz / 17 / 64 ≈ 44,117 Hz (main.c)', 'tm');
    },
  });

  // ---- waveform
  mk('wave', 600, 214, {
    build(g, el) {
      el.title = txt(g, 0, 14, '', 'tt');
      el.area = h('path', { class: 'wavefill' }, g);
      el.dc = h('line', { class: 'dcline' }, g);
      el.ax = h('g', {}, g);
      el.comb = h('g', { class: 'opt m-frames' }, g);
      el.hiF = h('rect', { class: 'opt m-frames hiframe', y: 26, height: 150 }, g);
      el.hiB = h('rect', { class: 'opt m-bytes hiframe', y: 26, height: 150 }, g);
      el.hiBt = txt(g, 0, 40, 'first UART frame: 1024 samples', 'tm opt m-bytes');
      el.hit = h('rect', { x: 50, y: 26, width: 540, height: 150, class: 'hit', 'data-h': 'wave' }, g);
    },
    update(el) {
      const X0 = 50, X1 = 590, Y0 = 30, Y1 = 172, n = R.env.min.length;
      const lo = Math.min(...R.env.min), hi = Math.max(...R.env.max);
      const y = v => Y1 - (v - lo) / (hi - lo) * (Y1 - Y0);
      const x = i => X0 + i / n * (X1 - X0);
      let d = '';
      R.env.max.forEach((v, i) => { d += (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1); });
      for (let i = n - 1; i >= 0; i--) d += 'L' + x(i + 1).toFixed(1) + ',' + y(R.env.min[i]).toFixed(1);
      el.area.setAttribute('d', d + 'Z');
      el.title.textContent = `The recording: ${fint(R.nSamples)} samples at ${fint(R.fileSr)} per second`;
      ['x1', 'x2', 'y1', 'y2'].forEach((k, i) => el.dc.setAttribute(k, [X0, X1, y(R.dc), y(R.dc)][i]));
      el.ax.replaceChildren();
      h('line', { x1: X0, y1: 180, x2: X1, y2: 180, class: 'axis' }, el.ax);
      const tx = t => X0 + t / R.dur * (X1 - X0);
      for (let t = 0; t <= Math.floor(R.dur); t++) { txt(el.ax, tx(t), 202, t + ' s', 'tm', 'middle'); h('line', { x1: tx(t), y1: 180, x2: tx(t), y2: 185, class: 'axis' }, el.ax); }
      txt(el.ax, X0 - 4, Y0 + 4, fint(hi), 'tm', 'end');
      txt(el.ax, X0 - 4, Y1 + 4, fint(lo), 'tm', 'end');
      txt(el.ax, X0 - 4, y(R.dc) + 4, fint(R.dc), 'tm', 'end');
      el.comb.replaceChildren();
      for (let f = 0; f < R.nFrames; f++) { const xx = tx(f * HOP / SR); h('line', { x1: xx, x2: xx, y1: 176, y2: 180, class: 'comb' }, el.comb); }
      const w = NFFT / SR;
      el.hiF.setAttribute('x', tx(R.frameT - w / 2)); el.hiF.setAttribute('width', Math.max(2, tx(w) - X0));
      el.hiB.setAttribute('x', X0); el.hiB.setAttribute('width', Math.max(2, tx(1024 / R.fileSr) - X0));
      el.hiBt.setAttribute('x', X0 + 8);
      el.geo = { X0, X1, n };
    },
  });
  HOV.wave = (t, e) => {
    const [fx_] = frac(t, e), i = Math.floor(fx_ * R.env.min.length);
    const per = R.nSamples / R.env.min.length;
    const s0 = Math.floor(i * per), s1 = Math.floor((i + 1) * per) - 1;
    return `t = ${(s0 / R.fileSr).toFixed(3)} s<br>samples ${fint(s0)}–${fint(s1)}<br>min ${fint(R.env.min[i])}, max ${fint(R.env.max[i])}<br><span style="opacity:.7">display slice ${i + 1} of 600</span>`;
  };

  // ---- UART bytes
  mk('bytes', 620, 196, {
    build(g, el) { el.in = h('g', {}, g); },
    update(el) {
      const g = el.in; g.replaceChildren();
      txt(g, 0, 14, 'What travels over USART3 (921,600 baud)', 'tt');
      let x = 0;
      const cell = (b, cls, tip, y = 50) => {
        h('rect', { x, y, width: 30, height: 28, rx: 4, class: cls, 'data-h': 'tip', 'data-tip': tip }, g);
        txt(g, x + 15, y + 18, hex2(b), 'tv', 'middle');
        x += 33;
      };
      txt(g, x, 42, 'START', 'tm');
      cell(0xBB, 'b-mark', 'START_MARKER byte 1 (0xBB)'); cell(0x66, 'b-mark', 'START_MARKER byte 2 (0x66)');
      x += 12; txt(g, x, 42, 'frame header', 'tm');
      cell(0xAA, 'b-head', 'sync byte 0xAA'); cell(0x55, 'b-head', 'sync byte 0x55');
      cell(0x00, 'b-head', 'length low byte: 1024 = 0x0400 → 00'); cell(0x04, 'b-head', 'length high byte: 1024 = 0x0400 → 04');
      x += 12; txt(g, x, 42, 'samples: low byte, then high byte', 'tm');
      for (let i = 0; i < 4; i++) {
        const v = R.first[i], u = v & 0xFFFF, lo = u & 255, hi = (u >> 8) & 255;
        const sx = x;
        cell(lo, 'b-data', `sample ${i} = ${fint(v)} = 0x${hex2(hi)}${hex2(lo)}<br>low byte first: ${hex2(lo)}`);
        cell(hi, 'b-data', `sample ${i} = ${fint(v)}<br>high byte: ${hex2(hi)}`);
        txt(g, sx + 31, 94, fint(v), 'tm', 'middle');
        x += 4;
      }
      txt(g, x + 2, 69, '…', 'tt');
      const y2 = 132;
      txt(g, 0, y2 + 18, `… 1020 more samples, then ${R.uartFrames - 1} more frames`, 'tm');
      txt(g, 0, y2 + 34, `(${R.uartFrames} frames × 1024 samples)`, 'tm');
      x = 494;
      txt(g, x, y2 - 8, 'STOP', 'tm');
      cell(0xCC, 'b-mark', 'STOP_MARKER byte 1 (0xCC)', y2); cell(0x77, 'b-mark', 'STOP_MARKER byte 2 (0x77)', y2);
      txt(g, 0, 188, 'The laptop glues the samples together and saves a .wav file.', 'tm');
    },
  });

  // ---- one frame
  mk('frame', 600, 196, {
    build(g, el) {
      el.title = txt(g, 0, 14, '', 'tt');
      el.sig = h('path', { class: 'sigline' }, g);
      el.hann = h('path', { class: 'hann' }, g);
      el.hann2 = h('path', { class: 'hann' }, g);
      h('line', { x1: 50, y1: 100, x2: 590, y2: 100, class: 'gridl' }, g);
      txt(g, 590, 192, 'dashed: Hann window (shape only)', 'tm', 'end');
      txt(g, 50, 192, 'sample 0', 'tm'); txt(g, 470, 192, '2047', 'tm');
      el.hit = h('rect', { x: 50, y: 28, width: 540, height: 146, class: 'hit', 'data-h': 'frame' }, g);
    },
    update(el) {
      const fr = R.frame, n = fr.length, amp = Math.max(...fr.map(Math.abs)) || 1;
      const x = i => 50 + i / (n - 1) * 540, y = v => 100 - v / amp * 66;
      el.sig.setAttribute('d', fr.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1)).join(''));
      let d1 = '', d2 = '';
      for (let i = 0; i <= 128; i++) {
        const wv = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 128);
        d1 += (i ? 'L' : 'M') + x(i / 128 * (n - 1)).toFixed(1) + ',' + y(wv * amp).toFixed(1);
        d2 += (i ? 'L' : 'M') + x(i / 128 * (n - 1)).toFixed(1) + ',' + y(-wv * amp).toFixed(1);
      }
      el.hann.setAttribute('d', d1); el.hann2.setAttribute('d', d2);
      el.title.textContent = `Frame #${R.frameIdx} of ${R.nFrames}: 2048 samples (46 ms) around t = ${R.frameT.toFixed(2)} s`;
    },
  });
  HOV.frame = (t, e) => {
    const [a] = frac(t, e), i = Math.floor(a * R.frame.length);
    return `sample ${i * 2} of 2048<br>value ${fx(R.frame[i], 4)} (scaled to −1…1)<br><span style="opacity:.7">every 2nd sample drawn</span>`;
  };

  // ---- spectrum of the frame (+ mel filters)
  const SX0 = 50, SX1 = 590, SY0 = 30, SY1 = 206;
  const sx = hz => SX0 + hz / 22050 * (SX1 - SX0);
  mk('spec', 600, 250, {
    build(g, el) {
      el.title = txt(g, 0, 14, '', 'tt');
      el.fill = h('path', { class: 'specfill' }, g);
      el.line = h('path', { class: 'specline' }, g);
      el.ax = h('g', {}, g);
      h('line', { x1: SX0, y1: SY1, x2: SX1, y2: SY1, class: 'axis' }, g);
      for (let k = 0; k <= 20000; k += 5000) {
        h('line', { x1: sx(k), y1: SY1, x2: sx(k), y2: SY1 + 5, class: 'axis' }, g);
        txt(g, sx(k), SY1 + 18, k ? (k / 1000) + ' kHz' : '0 Hz', 'tm', 'middle');
      }
      const mg = h('g', { class: 'opt m-mel' }, g);
      MODEL.filters.forEach(([lo, c, hi], k) => {
        const hl = k === 8 || k === 64 || k === 120;
        h('path', { d: `M${sx(lo)},${SY1} L${sx(c)},${SY0 + 34} L${sx(hi)},${SY1}`, class: hl ? 'tri-hl' : 'tri' }, mg);
        if (hl) txt(mg, sx(c), SY0 + 28, 'band ' + k, 'tm tacc halo', 'middle');
      });
      txt(mg, SX1, SY1 + 34, '128 mel filters (drawn at equal height)', 'tm', 'end');
      el.hit = h('rect', { x: SX0, y: SY0, width: SX1 - SX0, height: SY1 - SY0, class: 'hit', 'data-h': 'spec' }, g);
    },
    update(el) {
      const s = R.spec, hi = Math.max(...s) + 3, lo = Math.max(Math.min(...s), hi - 110);
      el.lo = lo; el.hi = hi;
      const y = v => SY1 - (clamp(v, lo, hi) - lo) / (hi - lo) * (SY1 - SY0);
      let d = '';
      s.forEach((v, i) => { d += (i ? 'L' : 'M') + sx(i * SR / NFFT).toFixed(1) + ',' + y(v).toFixed(1); });
      el.line.setAttribute('d', d);
      el.fill.setAttribute('d', d + `L${SX1},${SY1}L${SX0},${SY1}Z`);
      el.ax.replaceChildren();
      [hi, (hi + lo) / 2, lo].forEach(v => txt(el.ax, SX0 - 4, y(v) + 4, fx(v, 0), 'tm', 'end'));
      txt(el.ax, SX0 - 4, SY0 - 8, 'dB', 'tm', 'end');
      el.title.textContent = `Power spectrum of frame #${R.frameIdx}: 1025 frequency bins`;
    },
  });
  HOV.spec = (t, e) => {
    const [a] = frac(t, e), i = Math.round(a * 22050 / (SR / NFFT));
    const f = MODEL.filters, ks = [];
    f.forEach((q, k) => { const hz = i * SR / NFFT; if (hz > q[0] && hz < q[2]) ks.push(k); });
    return `bin ${i}: ${fint(i * SR / NFFT)} Hz<br>power ${fx(R.spec[i], 1)} dB` + (ks.length ? `<br>inside mel band${ks.length > 1 ? 's' : ''} ${ks.join(', ')}` : '');
  };

  // ---- one column of 128 mel energies (no text: it gets squeezed into the spectrogram)
  mk('melcol', 20, 384, {
    build(g, el) {
      el.cells = [];
      for (let k = 0; k < 128; k++)
        el.cells.push(h('rect', { x: 0, y: 384 - (k + 1) * 3, width: 20, height: 3, class: 'mcell', 'data-h': 'melc', 'data-k': k }, g));
    },
    update(el) { el.cells.forEach((c, k) => c.style.fill = seq((R.melCol[k] - R.melLo) / 80)); },
  });
  HOV.melc = t => {
    const k = +t.dataset.k, f = MODEL.filters[k];
    return `frame #${R.frameIdx}, mel band ${k}<br>${fint(f[0])}–${fint(f[2])} Hz (peak ${fint(f[1])} Hz)<br>${fx(R.melCol[k], 1)} dB`;
  };

  // ---- mel spectrogram heatmap
  mk('melspec', 580, 424, {
    build(g, el) {
      el.img = h('image', { x: 50, y: 8, height: 384, preserveAspectRatio: 'none', class: 'hm', 'data-h': 'mels' }, g);
      el.ax = h('g', {}, g);
      el.cur = h('rect', { y: 6, height: 388, width: 6, class: 'opt m-cursor cursor' }, g);
      [0, 32, 64, 96, 127].forEach(k => { const hz = MODEL.filters[k][1]; txt(g, 46, 8 + 384 - k * 3 + 3, hz < 1000 ? fint(hz) : (hz / 1000).toFixed(1) + 'k', 'tm', 'end'); });
      txt(g, 46, 418, 'Hz', 'tm', 'end');
    },
    update(el) {
      const cols = R.melspec[0].length;
      el.img.setAttribute('width', cols * 4);
      el.img.setAttribute('href', heatURL(R.melspec, v => seqA((v - R.melLo) / 80), true));
      el.ax.replaceChildren();
      for (let t = 0; t <= Math.floor(R.dur); t++) {
        const x = 50 + t * SR / HOP / R.melPool * 4;
        h('line', { x1: x, x2: x, y1: 394, y2: 399, class: 'axis' }, el.ax);
        txt(el.ax, x, 412, t + ' s', 'tm', 'middle');
      }
      const c = Math.floor(R.frameIdx / R.melPool);
      el.cur.setAttribute('x', 50 + c * 4 - 1);
    },
  });
  HOV.mels = (t, e) => {
    const [a, b] = frac(t, e), cols = R.melspec[0].length;
    const c = Math.floor(a * cols), k = 127 - Math.floor(b * 128), f = MODEL.filters[k];
    const f0 = c * R.melPool;
    return `frames ${f0}–${Math.min(f0 + R.melPool - 1, R.nFrames - 1)} (t ≈ ${(f0 * HOP / SR).toFixed(2)} s)<br>mel band ${k}, peak ${fint(f[1])} Hz<br>${R.melspec[k][c]} dB <span style="opacity:.7">(mean of 2 frames)</span>`;
  };

  // ---- DCT illustration
  const cosB = (k, n, N = 128) => Math.cos(Math.PI * k * (2 * n + 1) / (2 * N));
  mk('cos', 330, 410, {
    build(g, el) {
      txt(g, 0, 14, 'DCT: compare with cosine waves', 'tt');
      el.rows = [];
      for (let r = 0; r < 5; r++) {
        const y0 = 34 + r * 70;
        h('rect', { x: 0, y: y0, width: 220, height: 56, rx: 6, class: 'box' }, g);
        const p = h('path', { class: r ? 'hann' : 'sigline' }, g);
        if (r) {
          let d = '';
          for (let n = 0; n < 128; n++) d += (n ? 'L' : 'M') + (8 + n / 127 * 204).toFixed(1) + ',' + (y0 + 28 - cosB(r - 1, n) * 20).toFixed(1);
          p.setAttribute('d', d); p.style.strokeDasharray = 'none';
        }
        const lab = txt(g, 230, y0 + 24, r ? `cosine ${r - 1}` : 'mel dB', 'tm');
        const val = txt(g, 230, y0 + 42, '', 'ts tb');
        el.rows.push({ p, lab, val, y0 });
      }
      txt(g, 0, 400, '… then cosines 4–12; the rest are dropped', 'tm');
    },
    update(el) {
      const m = R.melCol, lo = Math.min(...m), hi = Math.max(...m);
      const r0 = el.rows[0];
      r0.p.setAttribute('d', m.map((v, n) => (n ? 'L' : 'M') + (8 + n / 127 * 204).toFixed(1) + ',' + (r0.y0 + 50 - (v - lo) / (hi - lo || 1) * 44).toFixed(1)).join(''));
      r0.val.textContent = `frame #${R.frameIdx}`;
      for (let k = 0; k < 4; k++) el.rows[k + 1].val.textContent = `→ c${k} = ${fx(R.mfccCol[k], 1)}`;
    },
  });

  // ---- the 13 MFCCs of one frame
  mk('mfcccol', 64, 312, {
    build(g, el) {
      el.cells = []; el.tx = [];
      for (let k = 0; k < 13; k++) {
        el.cells.push(h('rect', { x: 0, y: k * 24, width: 64, height: 24, class: 'cell', 'data-h': 'mfc', 'data-k': k }, g));
        el.tx.push(txt(g, 32, k * 24 + 16, '', 'tv txt-hide', 'middle'));
      }
    },
    update(el) {
      el.cells.forEach((c, k) => c.style.fill = div(R.mfccCol[k] / R.rowMax[k]));
      el.tx.forEach((t, k) => t.textContent = `c${k} ${fx(R.mfccCol[k], 1)}`);
    },
  });
  HOV.mfc = t => { const k = +t.dataset.k; return `frame #${R.frameIdx}, MFCC c${k}<br>${fx(R.mfccCol[k], 2)}`; };

  // ---- MFCC map
  mk('mfccmap', 580, 350, {
    build(g, el) {
      el.img = h('image', { x: 50, y: 8, height: 312, preserveAspectRatio: 'none', class: 'hm', 'data-h': 'mfm' }, g);
      for (let k = 0; k < 13; k++) txt(g, 46, 8 + k * 24 + 16, 'c' + k, 'tm', 'end');
      el.ax = h('g', {}, g);
      el.cur = h('rect', { y: 6, height: 316, width: 5, class: 'opt m-cursor cursor' }, g);
    },
    update(el) {
      el.img.setAttribute('width', R.nFrames * 2);
      el.img.setAttribute('href', heatURL(R.mfcc, (v, r) => divA(v / R.rowMax[r]), false));
      el.ax.replaceChildren();
      for (let t = 0; t <= Math.floor(R.dur); t++) {
        const x = 50 + t * SR / HOP * 2;
        h('line', { x1: x, x2: x, y1: 322, y2: 327, class: 'axis' }, el.ax);
        txt(el.ax, x, 340, t + ' s', 'tm', 'middle');
      }
      el.cur.setAttribute('x', 50 + R.frameIdx * 2 - 1.5);
    },
  });
  HOV.mfm = (t, e) => {
    const [a, b] = frac(t, e), f = Math.floor(a * R.nFrames), k = Math.floor(b * 13);
    return `frame #${f} (t = ${(f * HOP / SR).toFixed(2)} s)<br>MFCC c${k} = ${fx(R.mfcc[k][f], 1)}<br><span style="opacity:.7">color scaled per row</span>`;
  };

  // ---- the 52 features (raw, then scaled)
  const STATS = ['mean', 'std', 'max', 'min'];
  mk('feat', 340, 380, {
    build(g, el) {
      STATS.forEach((s, c) => txt(g, 50 + c * 70 + 33, 20, s, 'tm', 'middle'));
      el.cells = []; el.tx = [];
      for (let f = 0; f < 52; f++) {
        const r = f % 13, c = Math.floor(f / 13);
        el.cells.push(h('rect', { x: 50 + c * 70, y: 28 + r * 25, width: 66, height: 23, rx: 3, class: 'cell', 'data-h': 'feat', 'data-f': f }, g));
        el.tx.push(txt(g, 50 + c * 70 + 33, 28 + r * 25 + 16, '', 'tv', 'middle'));
      }
      for (let k = 0; k < 13; k++) txt(g, 44, 28 + k * 25 + 16, 'c' + k, 'tm', 'end');
      txt(g, 0, 370, '52 raw features (13 MFCCs × 4 stats)', 'tm opt m-raw');
      txt(g, 0, 370, 'z = (x − μ) / σ (μ, σ from 270 recordings)', 'tm opt m-scaled');
    },
    update(el) {
      const scaled = (el.g.dataset.mode || '').includes('scaled');
      el.cells.forEach((c, f) => c.style.fill = scaled ? div(R.scaled[f] / 3) : 'var(--zero)');
      el.tx.forEach((t, f) => t.textContent = scaled ? fx(R.scaled[f], 2) : fx(R.feat[f], Math.abs(R.feat[f]) >= 100 ? 0 : 1));
    },
    onMode(el) { el.update(el); },
  });
  HOV.feat = t => {
    const f = +t.dataset.f, r = f % 13, s = STATS[Math.floor(f / 13)];
    const scaled = (els.feat.g.dataset.mode || '').includes('scaled');
    const head = `feature ${f}: ${s} of c${r} over ${R.nFrames} frames`;
    if (!scaled) return `${head}<br>x = ${fx(R.feat[f], 3)}`;
    return `${head}<br>x = ${fx(R.feat[f], 2)}<br>μ = ${fx(MODEL.mean[f], 2)}, σ = ${fx(MODEL.scale[f], 2)}<br>z = ${fx(R.scaled[f], 3)}`;
  };

  // ---- PCA map of all clean recordings
  mk('pca', 450, 416, {
    build(g, el) {
      const pts = MODEL.pca.pts;
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const X = v => 44 + (v - x0) / (x1 - x0) * 396, Y = v => 396 - (v - y0) / (y1 - y0) * 340;
      el.X = X; el.Y = Y;
      h('rect', { x: 36, y: 48, width: 412, height: 356, rx: 6, class: 'box' }, g);
      txt(g, 242, 416, `PC1 (${(MODEL.pca.var[0] * 100).toFixed(1)}% of the spread)`, 'tm', 'middle');
      const yl = txt(g, 0, 0, `PC2 (${(MODEL.pca.var[1] * 100).toFixed(1)}%)`, 'tm', 'middle');
      yl.setAttribute('transform', 'translate(28,226) rotate(-90)');
      el.leg = h('g', {}, g);
      el.pts = pts.map(p => h('circle', {
        cx: X(p[0]).toFixed(1), cy: Y(p[1]).toFixed(1), r: 3.4, class: 'pt' + (p[4] ? ' ho' : ''), 'data-h': 'tip',
        'data-tip': `${p[5]}<br>really ${clsName(MODEL.order[p[2]])}, model says ${clsName(MODEL.order[p[3]])}<br>${p[4] ? 'held out (never seen in training)' : 'used for training'}`,
      }, g));
      el.lab = h('g', {}, g);
      el.me = h('circle', { r: 9, class: 'me' }, g);
      el.meT = txt(g, 0, 0, 'this recording', 'ts tb halo');
    },
    update(el) {
      const pts = MODEL.pca.pts;
      el.pts.forEach((c, i) => {
        const col = clsColor(pts[i][2]);
        c.style.fill = col; if (pts[i][4]) c.style.stroke = col;
      });
      el.leg.replaceChildren();
      MODEL.order.forEach((c, i) => {
        h('circle', { cx: 44 + i * 58, cy: 12, r: 5, fill: clsColor(i) }, el.leg);
        txt(el.leg, 52 + i * 58, 16, clsName(c), 'tm');
      });
      h('circle', { cx: 50, cy: 32, r: 3.4, fill: 'var(--muted)' }, el.leg); txt(el.leg, 58, 36, 'trained on', 'tm');
      h('circle', { cx: 140, cy: 32, r: 3.4, fill: 'var(--surface)', stroke: 'var(--muted)' }, el.leg); txt(el.leg, 148, 36, 'held out', 'tm');
      el.lab.replaceChildren();
      MODEL.order.forEach((c, i) => {
        const m = pts.filter(p => p[2] === i);
        const mx = m.reduce((s, p) => s + p[0], 0) / m.length, my = m.reduce((s, p) => s + p[1], 0) / m.length;
        txt(el.lab, el.X(mx), el.Y(my), clsName(c), 'ts tb halo', 'middle');
      });
      const me = pts.find(p => p[5] === R.id);
      if (me) {
        el.me.setAttribute('cx', el.X(me[0])); el.me.setAttribute('cy', el.Y(me[1]));
        el.meT.setAttribute('x', el.X(me[0]) + 12); el.meT.setAttribute('y', el.Y(me[1]) - 10);
      }
    },
  });

  // ---- one duel: w · x + b
  mk('wx', 486, 300, {
    build(g, el) { el.in = h('g', {}, g); },
    update(el) {
      const g = el.in; g.replaceChildren();
      const D = R.duel;
      txt(g, 0, 14, `One duel: ${clsName(D.a)} vs ${clsName(D.b)}`, 'tt');
      const grids = [
        ['w (learned weights)', D.w, Math.max(...D.w.map(Math.abs)), i => `weight w${i} = ${fx(D.w[i], 4)}`],
        ['x (scaled features)', R.scaled, 3, i => `x${i} = ${fx(R.scaled[i], 3)}`],
        ['w × x', D.prod, Math.max(...D.prod.map(Math.abs)), i => `w${i} × x${i} = ${fx(D.w[i], 3)} × ${fx(R.scaled[i], 3)}<br>= ${fx(D.prod[i], 4)}<br>${D.prod[i] > 0 ? 'pushes toward ' + clsName(D.a) : 'pushes toward ' + clsName(D.b)}`],
      ];
      grids.forEach(([title, vals, mx, tip], gi) => {
        const gx = 6 + gi * 112;
        txt(g, gx, 42, title, 'tm');
        vals.forEach((v, i) => {
          const r = i % 13, c = Math.floor(i / 13);
          h('rect', { x: gx + c * 18, y: 52 + r * 17, width: 17, height: 16, class: 'cell', fill: div(v / mx), 'data-h': 'tip', 'data-tip': `feature ${i}: ${STATS[c]} of c${r}<br>` + tip(i) }, g);
        });
      });
      txt(g, 92, 168, '×', 'tt', 'middle'); txt(g, 204, 168, '=', 'tt', 'middle');
      const bx = 330;
      h('rect', { x: bx, y: 52, width: 154, height: 150, rx: 8, class: 'box' }, g);
      txt(g, bx + 12, 76, 'add all 52:', 'tm');
      txt(g, bx + 142, 96, fx(D.sum, 3), 'ts tb', 'end');
      txt(g, bx + 12, 120, '+ bias b:', 'tm');
      txt(g, bx + 142, 140, fx(D.bias, 3), 'ts tb', 'end');
      h('line', { x1: bx + 12, x2: bx + 142, y1: 150, y2: 150, class: 'axis' }, g);
      txt(g, bx + 142, 170, '= ' + fx(D.sum + D.bias, 3), 'tt', 'end');
      txt(g, bx + 12, 192, D.val > 0 ? `> 0 → ${clsName(D.a)}` : `< 0 → ${clsName(D.b)}`, 'ts tb tacc');
      txt(g, 6, 290, `orange: pushes toward ${clsName(D.a)} (positive) · blue: toward ${clsName(D.b)}`, 'tm');
    },
  });

  // ---- 15 duels
  mk('duels', 470, 380, {
    build(g, el) { el.in = h('g', {}, g); },
    update(el) {
      const g = el.in; g.replaceChildren();
      const LE = MODEL.leClasses, mx = Math.max(...R.ovo.map(Math.abs));
      txt(g, 0, 14, '15 one-vs-one duels: w·x + b for each pair', 'tt');
      h('line', { x1: 290, x2: 290, y1: 26, y2: 26 + 15 * 22, class: 'zero' }, g);
      MODEL.pairs.forEach(([a, b], k) => {
        const y = 30 + k * 22, v = R.ovo[k], win = LE[v > 0 ? a : b];
        const mine = LE[a] === R.pred || LE[b] === R.pred;
        txt(g, 0, y + 14, `${clsName(LE[a])} vs ${clsName(LE[b])}`, mine ? 'ts tb' : 'ts');
        const len = Math.abs(v) / mx * 110;
        h('rect', { x: v > 0 ? 290 : 290 - len, y: y + 3, width: Math.max(1, len), height: 14, rx: 2, class: 'cell', fill: div(v / mx),
          'data-h': 'tip', 'data-tip': `${clsName(LE[a])} vs ${clsName(LE[b])}<br>w·x + b = ${fx(v, 3)}<br>${v > 0 ? 'positive' : 'negative'} → vote for ${clsName(win)}` }, g);
        txt(g, 128, y + 14, fx(v, 2), 'tm');
        txt(g, 412, y + 14, '→ ' + clsName(win), win === R.pred ? 'ts tb tacc' : 'ts');
      });
      txt(g, 180, 372, `← vote for 2nd class`, 'tm', 'middle'); txt(g, 400, 372, `vote for 1st class →`, 'tm', 'middle');
    },
  });

  // ---- bar lists (votes, probabilities)
  const barsEl = (name, title, vals) => mk(name, 350, 236, {
    build(g, el) {
      txt(g, 0, 14, title, 'tt');
      el.rows = MODEL.order.map((c, i) => {
        const y = 32 + i * 32;
        txt(g, 0, y + 15, clsName(c), 'ts');
        h('rect', { x: 64, y, width: 160, height: 20, rx: 3, class: 'track' }, g);
        const bar = h('rect', { x: 64, y, width: 160, height: 20, rx: 3, class: 'bar', 'data-h': 'tip' }, g);
        const t = txt(g, 232, y + 15, '', 'tv');
        return { bar, t, c };
      });
    },
    update(el) {
      el.rows.forEach(r => {
        const v = vals(r.c);
        r.bar.style.transform = `scaleX(${Math.max(0.004, v.frac)})`;
        r.bar.classList.toggle('win', r.c === R.pred);
        r.bar.dataset.tip = v.tip; r.t.textContent = v.text;
      });
    },
  });
  barsEl('votes', 'Votes (each class is in 5 duels)', c => {
    const i = MODEL.leClasses.indexOf(c), v = R.votes[i];
    return { frac: v / 5, text: `${v}   (score ${fx(R.ovr[i], 2)})`, tip: `${clsName(c)}: ${v} of 5 duels won<br>sklearn score ${fx(R.ovr[i], 3)} = votes + a small tie-breaker` };
  });
  barsEl('proba', 'Probabilities (predict_proba)', c => {
    const i = MODEL.leClasses.indexOf(c), p = R.proba[i];
    return { frac: p, text: (p * 100).toFixed(1) + '%', tip: `${clsName(c)}: ${(p * 100).toFixed(2)}%` };
  });

  // ---- confusion matrix
  mk('conf', 400, 390, {
    build(g, el) { el.in = h('g', {}, g); el.which = 'held'; },
    update(el) {
      const g = el.in; g.replaceChildren();
      const src = el.which === 'held' ? MODEL.heldOut : MODEL.otherDay;
      const cm = src.cm, O = MODEL.order;
      txt(g, 0, 14, el.which === 'held' ? `68 held-out recordings: ${src.correct} / ${src.n} right (${(src.correct / src.n * 100).toFixed(1)}%)`
        : `${src.n} recordings from Dec 6 (data/): ${src.correct} right (${(src.correct / src.n * 100).toFixed(1)}%)`, 'tt');
      txt(g, 82 + 3 * 50, 40, 'model says →', 'tm', 'middle');
      O.forEach((c, j) => txt(g, 82 + j * 50 + 24, 56, clsName(c), 'tm', 'middle'));
      cm.forEach((row, i) => {
        const tot = row.reduce((a, b) => a + b, 0) || 1;
        txt(g, 76, 64 + i * 46 + 27, clsName(O[i]), 'tm', 'end');
        row.forEach((v, j) => {
          const x = 82 + j * 50, y = 64 + i * 46;
          h('rect', { x, y, width: 48, height: 44, rx: 3, class: 'cell', fill: v ? seq(0.15 + 0.85 * v / tot) : 'var(--zero)', 'data-h': 'tip',
            'data-tip': `really ${clsName(O[i])}, model says ${clsName(O[j])}: ${v} of ${tot}` }, g);
          if (i === j) h('rect', { x: x + 1, y: y + 1, width: 46, height: 42, rx: 3, class: 'diag', 'pointer-events': 'none' }, g);
          txt(g, x + 24, y + 27, v, 'tv', 'middle');
        });
      });
      const yl = txt(g, 0, 0, '↑ really', 'tm', 'middle'); yl.setAttribute('transform', 'translate(12,200) rotate(-90)');
      txt(g, 0, 384, el.which === 'held' ? 'Computed for this page with the repo\'s saved model.' : 'A different session, never used in training.', 'tm');
    },
  });

  // ---- command byte table
  mk('cmd', 400, 256, {
    build(g, el) { el.in = h('g', {}, g); },
    update(el) {
      const g = el.in; g.replaceChildren();
      txt(g, 0, 14, 'Class → command byte → LED pattern', 'tt');
      MODEL.order.forEach((c, i) => {
        const y = 32 + i * 34;
        if (c === R.pred) h('rect', { x: -4, y: y - 3, width: 404, height: 30, rx: 6, class: 'rowhl' }, g);
        txt(g, 0, y + 17, clsName(c), c === R.pred ? 'ts tb' : 'ts');
        txt(g, 76, y + 17, '0x' + hex2(CMD[c]), 'tv');
        const [n, col] = LED[c];
        for (let k = 0; k < 15; k++)
          h('rect', { x: 128 + k * 18, y: y + 4, width: 14, height: 16, rx: 3, fill: k < n ? rgb(col) : 'var(--empty)', stroke: 'var(--frame)', 'stroke-width': .6,
            'data-h': 'tip', 'data-tip': `${clsName(c)}: LED_ShowBar(${n}, ${col.join(', ')})` }, g);
      });
      txt(g, 0, 246, 'From CLASS_TO_CMD (Python) and LED_Service() (main.c)', 'tm');
    },
  });

  // ---- WS2812B bits
  mk('ledbits', 620, 226, {
    build(g, el) { el.in = h('g', {}, g); },
    update(el) {
      const g = el.in; g.replaceChildren();
      const [, col, name] = LED[R.pred];
      const grb = [col[1], col[0], col[2]];
      txt(g, 0, 14, `Each lit LED gets 24 bits: G, R, B = ${grb.join(', ')} (${name})`, 'tt');
      const bits = [];
      grb.forEach(v => { for (let b = 7; b >= 0; b--) bits.push((v >> b) & 1); });
      const sw = 24, x0 = 40;
      ['green', 'red', 'blue'].forEach((s, i) => txt(g, x0 + i * 8 * sw + 4 * sw, 36, s + ' (8 bits)', 'tm', 'middle'));
      let d = `M${x0},150`;
      bits.forEach((b, i) => {
        const x = x0 + i * sw;
        h('rect', { x: x + 1, y: 44, width: sw - 2, height: 24, rx: 3, class: b ? 'b-head' : 'b-data', 'data-h': 'tip',
          'data-tip': `bit ${i}: ${b} → high for ${b ? 38 : 19} of 60 timer ticks (${b ? '0.79' : '0.40'} µs)` }, g);
        txt(g, x + sw / 2, 61, b, 'tv', 'middle');
        const hiW = sw * (b ? 38 : 19) / 60;
        d += `L${x},100L${x + hiW},100L${x + hiW},150L${x + sw},150`;
      });
      h('path', { d, class: 'pulse-line' }, g);
      txt(g, x0 - 4, 104, 'high', 'tm', 'end'); txt(g, x0 - 4, 154, 'low', 'tm', 'end');
      txt(g, 0, 180, 'TIM2 at 48 MHz: 60 ticks per bit = 1.25 µs.', 'tm');
      txt(g, 0, 198, '1 → high for 38 ticks · 0 → high for 19 ticks', 'tm');
      txt(g, 0, 216, `× ${LED[R.pred][0]} lit LEDs (the other ${15 - LED[R.pred][0]} get zeros), then 80 low slots to latch.`, 'tm');
    },
  });
}

// ------------------------------------------------------------------ scenes (one per step)
function sc(list, modes = {}, arrows = [], extra) {
  const p = {};
  list.forEach(([n, r, max]) => { p[n] = fit(n, r, max); });
  return { p, modes, arrows, ...(extra || {}) };
}
const SCENES = {
  hardware() {
    const R0 = content();
    if (wide) {
      const [a, b, c] = split(R0, [0.8, 1, 1], 'h', 40);
      return sc([['bottle', a, 1.2], ['chip', b, 1.2], ['laptop', c, 1.2]], { bottle: 'idle' },
        [['bottle', 'chip', 'tap + sound', -14], ['chip', 'bottle', 'LED colors', 14], ['chip', 'laptop', 'PCM over UART', -14], ['laptop', 'chip', '1 byte', 14]]);
    }
    const [a, b] = split(R0, [0.42, 0.58], 'h', 26);
    const [b1, b2] = split(b, [1, 1], 'v', 26);
    return sc([['bottle', a], ['chip', b1], ['laptop', b2]], { bottle: 'idle' },
      [['bottle', 'chip', '', 0], ['chip', 'laptop', 'UART', -16], ['laptop', 'chip', '1 byte', 16]]);
  },
  tap() {
    const [a, b] = wide ? split(content(), [0.55, 1.5], 'h', 30) : split(content(), [0.5, 0.5], 'v', 14);
    return sc([['bottle', wide ? a : b], ['timeline', wide ? b : a]], { bottle: 'tap rec', timeline: 'play' });
  },
  pdm() {
    const [a, b] = split(content(), [0.85, 1], 'v', 30);
    return sc([['pdm', a, 1.3], ['wave', b, 1.3]], {}, [['pdm', 'wave', '', 0]]);
  },
  wave() { return sc([['wave', content(), 1.6]]); },
  bytes() {
    const [a, b] = split(content(), [1, 0.95], 'v', 30);
    return sc([['wave', a, 1.3], ['bytes', b, 1.3]], { wave: 'bytes' }, [['wave', 'bytes', '', 0]]);
  },
  frames() {
    const [a, b] = split(content(), [1, 0.95], 'v', 30);
    return sc([['wave', a, 1.3], ['frame', b, 1.3]], { wave: 'frames' }, [['wave', 'frame', 'one frame', 0]]);
  },
  fft() {
    const [a, b] = split(content(), [0.75, 1], 'v', 30);
    return sc([['frame', a, 1.2], ['spec', b, 1.3]], {}, [['frame', 'spec', 'FFT', 0]]);
  },
  mel() {
    const R0 = content();
    const [a, b] = split(R0, [1, wide ? 0.09 : 0.11], 'h', wide ? 50 : 30);
    const s = fit('spec', a, 1.4);
    const p = { spec: s };
    const th = 176 * s.s;   // the spectrum's plot height
    p.melcol = { x: b.x + (b.w - 22) / 2, y: s.y + 30 * s.s, sx: 22 / 20, sy: th / 384 };
    return { p, modes: { spec: 'mel' }, arrows: [['spec', 'melcol', '', 0]] };
  },
  melspec() {
    const m = fit('melspec', content(), 1.5);
    const c = Math.floor(R.frameIdx / R.melPool);
    return { p: { melspec: m, melcol: { x: m.x + m.s * (50 + c * 4), y: m.y + m.s * 8, sx: m.s * 4 / 20, sy: m.s } }, modes: { melspec: 'cursor' } };
  },
  dct() {
    const R0 = content();
    const [a, b, c] = split(R0, [0.16, 1, 0.3], 'h', wide ? 40 : 22);
    const cs = fit('cos', b, 1.3);
    const th = Math.min(a.h * 0.85, 384 * 1.1);
    const mc = fit('mfcccol', c, 1.3);
    return {
      p: { melcol: { x: a.x + (a.w - 22) / 2, y: a.y + (a.h - th) / 2, sx: 22 / 20, sy: th / 384 }, cos: cs, mfcccol: mc },
      modes: {}, arrows: [['melcol', 'cos', '', 0], ['cos', 'mfcccol', 'keep 13', 0]],
    };
  },
  mfccmap() {
    const m = fit('mfccmap', content(), 1.5);
    return { p: { mfccmap: m, mfcccol: { x: m.x + m.s * (50 + R.frameIdx * 2), y: m.y + m.s * 8, sx: m.s * 2 / 64, sy: m.s } }, modes: { mfccmap: 'cursor', mfcccol: 'map' } };
  },
  stats() {
    const [a, b] = auto(wide ? [1.5, 1] : [1, 1.1]);
    return sc([['mfccmap', a], ['feat', b, 1.3]], { feat: 'raw' }, [['mfccmap', 'feat', 'mean, std, max, min', 0]]);
  },
  scaler() { return sc([['feat', content(), 1.4]], { feat: 'scaled' }); },
  pca() {
    if (!wide) return sc([['pca', content(), 1.4]]);
    const [a, b] = split(content(), [0.7, 1.3], 'h', 50);
    return sc([['feat', a], ['pca', b, 1.4]], { feat: 'scaled' }, [['feat', 'pca', 'PCA, to look at', 0]]);
  },
  duel() { return sc([['wx', content(), 1.5]]); },
  duels() {
    const [a, b] = auto(wide ? [1.45, 1] : [1.3, 0.75]);
    return sc([['duels', a, 1.3], ['votes', b, 1.3]], {}, [['duels', 'votes', 'count votes', 0]]);
  },
  proba() {
    const [a, b] = auto([1, 1]);
    return sc([['votes', a, 1.3], ['proba', b, 1.3]]);
  },
  conf() { return sc([['conf', content(), 1.4]]); },
  cmd() {
    const [a, b] = split(content(), [1.4, 0.6], 'h', wide ? 50 : 20);
    return sc([['cmd', a, 1.3], ['bottle', b, 1.2]], { bottle: 'result' }, [['cmd', 'bottle', 'USART3', 0]]);
  },
  leds() {
    const [a, b] = wide ? split(content(), [2, 0.6], 'h', 40) : split(content(), [0.45, 0.55], 'v', 16);
    return sc([['ledbits', a, 1.3], ['bottle', b, 1.2]], { bottle: 'result' }, [['ledbits', 'bottle', '', 0]]);
  },
  overview() {
    const R0 = content();
    const names = ['wave', 'melspec', 'mfccmap', 'feat', 'votes', 'bottle'];
    const rows = wide ? split(R0, [1, 1], 'v', 34) : split(R0, [1, 1, 1], 'v', 22);
    const cells = [];
    rows.forEach(r => split(r, wide ? [1, 1, 1] : [1, 1], 'h', wide ? 44 : 26).forEach(c => cells.push(c)));
    const order = wide ? [0, 1, 2, 5, 4, 3] : [0, 1, 3, 2, 4, 5];
    const p = {};
    names.forEach((n, i) => { p[n] = fit(n, cells[order[i]], 1); });
    const arrows = names.slice(1).map((n, i) => [names[i], n, '', 0]);
    return { p, modes: { bottle: 'result', feat: 'scaled' }, arrows, lit: names };
  },
};

// ------------------------------------------------------------------ controls
function picker() {
  const box = document.createElement('label');
  box.className = 'hs-ctl';
  box.innerHTML = '<span>Recording</span>';
  const sel = document.createElement('select');
  sel.setAttribute('aria-label', 'Recording to follow');
  MODEL.recs.forEach(r => {
    const o = document.createElement('option');
    o.value = r.id;
    o.textContent = `${clsName(r.cls)} bottle` + (r.cls !== r.pred ? ' (a miss)' : '');
    sel.append(o);
  });
  sel.value = R.id;
  sel.onchange = () => setRec(sel.value);
  box.append(sel);
  CTX.controls.append(box);
}
function setRec(id) {
  R = RECS[id];
  CTX.data.rec = id;
  updateAll();
  if (curScene) applyScene(curScene, true);
  fillVars();
}

// ------------------------------------------------------------------ steps
function step(title, body, scene, extra) {
  return {
    title, body,
    enter(ctx) {
      clearTimers();
      picker();
      applyScene(SCENES[scene], ctx.from >= 0);
      fillVars();
      extra?.(ctx);
    },
    leave() { clearTimers(); },
  };
}
const V = n => `<b data-v="${n}"></b>`;

function confToggle(ctx) {
  const box = document.createElement('div');
  box.className = 'hs-ctl';
  [['held', 'Held out (Dec 7)'], ['other', 'Other day (Dec 6)']].forEach(([k, s]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = s; b.classList.toggle('on', els.conf.which === k);
    b.onclick = () => { els.conf.which = k; els.conf.update(els.conf); [...box.children].forEach(x => x.classList.toggle('on', x === b)); };
    box.append(b);
  });
  ctx.controls.append(box);
}
function playButton(ctx) {
  const box = document.createElement('div');
  box.className = 'hs-ctl';
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = '▶ Play';
  b.onclick = () => play();
  box.append(b);
  ctx.controls.append(box);
  later(play, 900);
}
function play() {
  clearTimers();
  const boxes = [...gOver.querySelectorAll('.litbox')];
  const names = ['wave', 'melspec', 'mfccmap', 'feat', 'votes', 'bottle'];
  setMode(els.bottle, 'rec');
  boxes.forEach(b => b.classList.remove('lit'));
  names.forEach((n, i) => later(() => {
    boxes.forEach(b => b.classList.toggle('lit', b.dataset.lit === n));
    if (n === 'bottle') setMode(els.bottle, 'result');
  }, 300 + i * 750));
  later(() => boxes.forEach(b => b.classList.remove('lit')), 300 + names.length * 750 + 900);
}

const CHAPTERS = [
  { title: 'The bottle and the tap', steps: [
    step('HydroSense at a glance', `<p>You can't see the water in an opaque bottle, but you can hear it. HydroSense listens to the bottle being shaken and lights up a strip of 15 LEDs to show how full it is: <b>empty, 100, 300, 500, 700 or 900 ml</b>.</p>
<p>The work is split in two. An STM32 board does the listening and the lights; a laptop runs the machine learning. They talk over a serial cable.</p>
<p>We'll follow one real recording from the repo, ${V('file')}, all the way through. Use the menu under the picture to switch recordings at any step.</p>`, 'hardware'),
    step('A tap starts it', `<p>An IMU (the LSM6DSO32 motion sensor) is set up to detect a single tap. When it feels one, it raises an interrupt pin and the board sets a flag.</p>
<p>The board then waits <b>1000 ms</b> so you can start shaking, records for <b>3 seconds</b>, and blinks the LEDs orange every 100 ms while it listens.</p>
<p class="note">The firmware's log message says "Recording in 0.5 sec", but the code (PRE_RECORD_DELAY_MS) waits 1000 ms.</p>`, 'tap'),
    step('Vibration becomes numbers', `<p>The microphone is a VPU (voice pickup) sensor that feels vibration through the bottle. It outputs <b>PDM</b>: one bit per clock tick, where the share of 1s tracks the sound.</p>
<p>The STM32's DFSDM block turns those bits into numbers. The mic clock is 48 MHz ÷ 17 ≈ 2.82 MHz, and a sinc³ filter averages each 64 bits into one sample: 2.82 MHz ÷ 64 = <b>44,117 samples per second</b>.</p>
<p>DMA copies samples into two 1024-sample halves of a buffer, so the CPU can send one half while the other fills.</p>`, 'pdm'),
    step('Three seconds of sloshing', `<p>Here is the real recording: ${V('n')} samples, about ${V('dur')} s. Each sample is a 16-bit integer between −32,768 and 32,767.</p>
<p>The picture shows the min and max of 600 slices (downsampled for drawing). Hover it for exact values.</p>
<p>The wave isn't centered on 0: the mic adds an offset of about ${V('dc')} (dashed line). That's normal and doesn't hurt what comes next.</p>`, 'wave'),
    step('Over the wire', `<p>The board streams the samples to the laptop over USART3 at 921,600 baud. Markers tell the Python script where a recording starts and stops: <code>BB 66</code> to start, <code>CC 77</code> to stop.</p>
<p>In between come frames: <code>AA 55</code>, a 2-byte length, then the samples. Numbers go <b>low byte first</b> (little-endian), so the length 1024 = 0x0400 is sent as <code>00 04</code>.</p>
<p>This recording arrived as ${V('uartFrames')} frames of 1024 samples. The laptop glues them together and saves a .wav file. Hover any byte.</p>`, 'bytes'),
  ] },
  { title: 'Sound to a picture', steps: [
    step('Slice it into frames', `<p>Sound changes over time, so the laptop studies it in short slices. First, <code>librosa.load(sr=44100)</code> resamples the file from 44,117 to 44,100 samples per second and scales it to −1…1.</p>
<p>Each <b>frame</b> is 2048 samples (46 ms). A new one starts every 512 samples (11.6 ms), so they overlap. This clip makes ${V('nFrames')} frames (the tick marks).</p>
<p>We'll follow frame ${V('frameIdx')}, the loudest one, at ${V('frameT')}. The dashed curve is the Hann window that fades the frame's edges before the next step.</p>`, 'frames'),
    step('Which pitches are inside?', `<p>An <b>FFT</b> (fast Fourier transform) splits the frame into pure tones and says how strong each one is. 2048 samples give 1025 frequency bins from 0 to 22,050 Hz, each 21.5 Hz apart.</p>
<p>Height is power in <b>decibels</b>, a log scale: +10 dB means 10 times more power.</p>
<p>How much water is in the bottle changes which frequencies are strongest. That's the clue the classifier will learn to read.</p>`, 'fft'),
    step('The mel filterbank', `<p>Our ears tell low pitches apart much more finely than high ones. A <b>mel filterbank</b> copies that: 128 triangle filters, narrow at low frequencies and wide at high ones.</p>
<p>Band 8 covers ${V('band8')} (${V('w8')} Hz wide); band 120 covers ${V('band120')} (${V('w120')} Hz wide).</p>
<p>Each filter adds up the power under its triangle. The 128 results, in dB, form the colored column on the right: low bands at the bottom, brighter = more energy.</p>`, 'mel'),
    step('Every frame: a mel spectrogram', `<p>Do the same for all ${V('nFrames')} frames and stand the columns side by side. That's a <b>mel spectrogram</b>: time goes right, pitch goes up, brightness is energy. Our frame's column slides into its slot.</p>
<p>The drawing averages pairs of frames so it fits; hover any cell for its value. librosa also floors everything at 80 dB below the loudest cell, which is why the darkest cells are all about ${V('melLo')} dB.</p>`, 'melspec'),
  ] },
  { title: 'MFCC', steps: [
    step('13 numbers per frame', `<p>128 numbers per frame is a lot, and neighboring bands are very similar. The <b>DCT</b> (discrete cosine transform) summarizes the column's shape by comparing it with cosine waves.</p>
<p>Cosine 0 is flat, so c0 measures overall loudness. Cosine 1 is a half wave, so c1 says "more low or more high pitch?". Higher ones catch finer wiggles.</p>
<p>Keeping only the first 13 gives the <b>MFCCs</b> (mel-frequency cepstral coefficients). For frame ${V('frameIdx')}: c0 = ${V('c0')}, c1 = ${V('c1')}.</p>`, 'dct'),
    step('MFCCs for the whole clip', `<p>Doing that for every frame gives a grid of 13 rows × ${V('nFrames')} frames. Each row is one coefficient changing over time; our frame's 13 numbers drop into their column.</p>
<p>Colors are scaled row by row (c0 is far bigger than the rest), orange above zero and blue below. Hover a cell for its exact value.</p>`, 'mfccmap'),
    step('52 numbers per recording', `<p>The classifier needs the same number of inputs for every recording. So <code>extract_features()</code> squeezes each row over time into 4 numbers: <b>mean, standard deviation, max and min</b>.</p>
<pre>features.extend(mfccs_mean)
features.extend(mfccs_std)
features.extend(mfccs_max)
features.extend(mfccs_min)</pre>
<p>13 × 4 = <b>52 numbers</b>: this recording's fingerprint. Hover a cell to see which row and statistic it came from.</p>`, 'stats'),
  ] },
  { title: 'The SVM decides', steps: [
    step('StandardScaler', `<p>The 52 numbers live on very different scales: c0's mean is ${V('x0')}, while many others sit near 0. Big numbers would drown out small ones.</p>
<p><b>StandardScaler</b> fixes that. For each feature it subtracts the average μ and divides by the spread σ, both measured on the 270 training recordings: <code>z = (x − μ) / σ</code>.</p>
<p>Now 0 means "typical", +2 means "unusually high". For c0's mean: (${V('x0')} − ${V('mu0')}) / ${V('sd0')} = ${V('z0')}. Orange is above average, blue below.</p>`, 'scaler'),
    step('Where does it land?', `<p>Each recording is now a point in 52-dimensional space. We can't draw that, so <b>PCA</b> finds the two directions in which the 338 clean recordings spread out most and flattens them onto a page.</p>
<p>Recordings with the same water level clump together, and neighbors like 300 and 500 ml overlap a bit. The big circle is our recording.</p>
<p class="note">This map is only for us to look at. The SVM itself works with all 52 numbers.</p>`, 'pca'),
    step('One duel: w·x + b', `<p>A <b>linear SVM</b> separates two classes with a flat boundary. Its whole decision is one sum: multiply each scaled feature by a learned weight w, add them up, then add a bias b. Positive means the first class wins; negative, the second.</p>
<p>The closest duel for our recording, ${V('duelA')} vs ${V('duelB')}: the 52 products add up to ${V('duelSum')}, plus b = ${V('duelB0')}, giving ${V('duelVal')}. So ${V('duelWin')} wins.</p>`, 'duel'),
    step('15 duels, one vote each', `<p>With 6 classes, the saved model (scikit-learn <code>SVC(kernel='linear', C=1)</code>) holds one duel for every pair: 6 × 5 ÷ 2 = <b>15 linear classifiers</b>. This is called <b>one-vs-one</b>.</p>
<p>Each duel gives one vote, and the class with the most votes wins. Here ${V('pred')} wins with ${V('votes')} of 5 possible votes.</p>
<p class="note">The model's setting <code>decision_function_shape='ovr'</code> only turns those votes into one score per class (votes plus a small tie-breaker). Underneath it is still one-vs-one.</p>`, 'duels'),
    step('How sure is it?', `<p><code>tap_classifier.py</code> also prints a probability per class. With <code>probability=True</code>, scikit-learn fits a small curve to each duel's score during training (Platt scaling) and combines the 15 results.</p>
<p>For this recording: ${V('pred')} at ${V('predP')}.</p>
<p>Votes and probabilities are computed differently, so on close calls they can disagree. The LEDs follow the votes (<code>model.predict</code>).</p>`, 'proba'),
    step('How often is it right?', `<p>The saved model was trained on 270 of the 338 recordings in <code>clean_data/</code>. We worked out exactly which 270 (its saved support vectors and the scaler's stored averages pin them down), so the other <b>68 were never seen</b>.</p>
<p>On those 68 it gets <b>63 right (92.6%)</b>. The mistakes are between neighbors: 300 vs 500, 500 vs 700, 900 vs 700.</p>
<p>Switch to "Other day" for the 343 recordings in <code>data/</code> from Dec 6: only 87 right (25.4%). Sound from another session looks different to the model, so testing on a new day matters.</p>
<p class="note">Both numbers were computed for this page with the repo's saved model. The notebook's printed scores (100% test, 94.7% cross-validation) come from an earlier 5-class run without 700 ml.</p>`, 'conf', confToggle),
  ] },
  { title: 'Back to the LEDs', steps: [
    step('One byte back', `<p>The laptop turns the class name into a single byte and sends it back over the same serial link: empty → <code>0x20</code>, 100 ml → <code>0x21</code>, up to 900 ml → <code>0x25</code>.</p>
<p>On the board, a receive interrupt (<code>HAL_UART_RxCpltCallback</code>) reads the byte and switches the LED state. Our recording: ${V('pred')} → ${V('byte')}.</p>`, 'cmd'),
    step('Lighting the strip', `<p>WS2812B LEDs read their color as 24 bits, in the order <b>green, red, blue</b>. The board makes the bits with a timer (TIM2) whose pulse widths are fed by DMA.</p>
<p>Each bit lasts 60 timer ticks (1.25 µs at 48 MHz). A 1 stays high for 38 ticks, a 0 for 19. After all 15 LEDs, 80 low slots tell the strip to show the new colors.</p>
<p>${V('pred')} lights ${V('ledN')} LEDs ${V('ledC')}. When idle, the strip glows blue.</p>`, 'leds'),
    step('The whole pipeline', `<p>End to end: ${V('dur')} s of sloshing → ${V('nFrames')} frames → 128 mel bands → 13 MFCCs → 52 numbers → scaled → 15 duels → ${V('pred')} → ${V('byte')} → ${V('ledN')} LEDs.</p>
<p>Press Play to watch it flow, and pick other recordings to see the prediction change. One of them is a real mistake: a 300 ml bottle the model calls 500 ml.</p>
<p>This recording was really ${V('cls')}: ${V('heldNote')}.</p>`, 'overview', playButton),
  ] },
];

// ------------------------------------------------------------------ start
async function init(ctx) {
  CTX = ctx;
  stageEl = document.getElementById('stage');
  ctx.stage.innerHTML = '<p class="muted" style="padding:16px">Loading the recordings…</p>';
  MODEL = await (await fetch(new URL('model.json', DATA))).json();
  const all = await Promise.all(MODEL.recs.map(r => fetch(new URL(`rec_${r.id}.json`, DATA)).then(x => x.json())));
  all.forEach(r => { RECS[r.id] = derive(r); });
  R = RECS[MODEL.hero];
  ctx.data.rec = R.id;
  ctx.stage.innerHTML = '';
  svg = h('svg', { id: 'hs-svg', role: 'img', 'aria-label': 'HydroSense pipeline drawing' }, ctx.stage);
  gEls = h('g', {}, svg);
  gOver = h('g', {}, svg);
  readPalette();
  measure();
  defineElements();
  gEls.append(els.melcol.g, els.mfcccol.g);   // keep the single columns above the heatmaps
  updateAll();
  svg.addEventListener('pointermove', onHover);
  svg.addEventListener('pointerdown', onHover);
  svg.addEventListener('pointerleave', () => { ctx.tip.hide(); if (hlNode) { hlNode.classList.remove('hl'); hlNode = null; } });
  window.addEventListener('themechange', () => { readPalette(); updateAll(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { readPalette(); updateAll(); });
  let rt = null;
  new ResizeObserver(() => {
    clearTimeout(rt);
    rt = setTimeout(() => { measure(); if (curScene) applyScene(curScene, false); }, 60);
  }).observe(stageEl);
}
function measure() {
  W = Math.max(280, stageEl.clientWidth);
  H = Math.max(240, stageEl.clientHeight);
  wide = W >= 640 && W / H > 1.05;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.classList.toggle('narrow', W < 560);
}

createStepper({
  slug: 'hydrosense',
  title: 'HydroSense, under the hood',
  credit: 'Team project, EECS 373 (Winter 2025): Yunjong (Kevin) Na, Saideep Kasipathy, Min Park',
  repo: 'https://github.com/saideepk444/HydroSense',
  chapters: CHAPTERS,
  init,
}).catch(err => {
  console.error(err);
  const s = document.getElementById('stage-inner');
  if (s) s.innerHTML = '<p style="padding:16px">Could not load the data for this page. Try reloading.</p>';
});
