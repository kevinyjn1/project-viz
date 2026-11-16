// Checkpoint Champion, under the hood.
// Every number drawn here comes from data/*.json, written by tools/lol-predictor/precompute.py,
// which re-runs the notebook's final pipeline (LASSO logistic regression) on data/lol_2024.csv.
import { createStepper, css } from '../assets/stepper.js';

const NS = 'http://www.w3.org/2000/svg';
const REPO = 'https://github.com/MrTuge123/League_of_Legends_Analysis';
const load = f => fetch(new URL('./data/' + f, import.meta.url)).then(r => {
  if (!r.ok) throw new Error(f + ': HTTP ' + r.status);
  return r.json();
});

let M, GM, TE;
try {
  [M, GM, TE] = await Promise.all([load('model.json'), load('game.json'), load('test.json')]);
} catch (err) {
  document.body.innerHTML = `<p style="padding:16px;font-family:sans-serif">Could not load the data files (${err.message}).</p>`;
  throw err;
}

// ---------------------------------------------------------------- formatting
const MINUS = '−';
const fx = (v, d = 2) => { const s = Math.abs(v).toFixed(d); return (v < 0 && +s !== 0 ? MINUS : '') + s; };
const sg = (v, d = 2) => (v > 0 && +v.toFixed(d) !== 0 ? '+' : '') + fx(v, d);
const int = v => (v < 0 && Math.round(v) !== 0 ? MINUS : '') + Math.round(Math.abs(v)).toLocaleString('en-US');
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const n1 = v => Math.round(v * 10) / 10;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------- data shortcuts
const NUM = M.numeric;                                   // the 10 numeric inputs
const IDX = Object.fromEntries(NUM.map((c, i) => [c, i]));
const W_OF = c => M.coef[IDX[c]];
const TEAMS = GM.teams;                                  // [Bilibili Gaming (Blue), T1 (Red)]
const SHORT = ['BLG', 'T1'];
const T1 = TEAMS[1];
const LEAGUES = M.leagues;
const LIDX = T1.league_index;
const LEAGUE_W = M.coef.slice(NUM.length);
const LEAGUE_NONZERO = LEAGUE_W.filter(v => v !== 0).length;
const NONZERO = M.coef.filter(v => v !== 0).length;
const FEAT9 = NUM.filter(c => c !== 'gamelength');
const ORDER = NUM.slice().sort((a, b) => Math.abs(W_OF(b)) - Math.abs(W_OF(a)));
const cnt = M.counts, met = M.metrics;
const [[TP, FN], [FP, TN]] = met.cm;
const LEARN = M.frames;

const DESC = {
  goldat25: 'total gold the team had by 25:00', xpat25: 'total experience (XP) by 25:00',
  csat25: 'creep score: minions and monsters killed by 25:00', killsat25: 'champion kills by 25:00',
  deathsat25: 'deaths by 25:00', assistsat25: 'assists by 25:00',
  golddiffat25: 'gold lead: goldat25 − opp_goldat25', xpdiffat25: 'XP lead: xpat25 − opp_xpat25',
  csdiffat25: 'CS lead: csat25 − opp_csat25', side: 'Blue or Red side of the map',
  side_binary: 'side as a number: Blue = 0, Red = 1', gamelength: 'final length of the game in seconds',
  league: 'which league or tournament the game was in',
};
for (const k of ['goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25', 'assistsat25'])
  DESC['opp_' + k] = 'the opponent’s ' + DESC[k].replace(/^total /, '');

// ---------------------------------------------------------------- colors
const TOK = ['surface', 'ink', 'muted', 'line', 'accent', 'flow', 'stage', 'neg', 'pos', 'zero', 'empty', 'cell-line', 'frame', 'good', 'bad'];
const C = {};
function readColors() { for (const t of TOK) C[t.replace('-', '_')] = css('--' + t) || '#888888'; }
const PC = new Map();
function rgb(s) {
  if (PC.has(s)) return PC.get(s);
  let v = null, m; const t = String(s).trim();
  if (t[0] === '#') {
    let h = t.slice(1); if (h.length === 3) h = h.split('').map(c => c + c).join('');
    if (h.length >= 6) v = [0, 2, 4].map(i => parseInt(h.substr(i, 2), 16));
  } else if ((m = t.match(/rgba?\(([^)]+)\)/))) v = m[1].split(/[\s,/]+/).filter(Boolean).slice(0, 3).map(Number);
  PC.set(s, v); return v;
}
const isColor = s => typeof s === 'string' && (s[0] === '#' || s.startsWith('rgb'));
function mix(a, b, t) {
  const A = rgb(a), B = rgb(b); if (!A || !B) return t < .5 ? a : b;
  return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`;
}
function lum(c) {
  const v = rgb(c); if (!v) return .5;
  const [r, g, b] = v.map(x => { x /= 255; return x <= .03928 ? x / 12.92 : Math.pow((x + .055) / 1.055, 2.4); });
  return .2126 * r + .7152 * g + .0722 * b;
}
const textOn = f => lum(f) > .2 ? '#16202B' : '#EEF2F6';
const div = (v, max) => mix(C.zero, v < 0 ? C.neg : C.pos, Math.pow(Math.min(1, Math.abs(v) / max), .8));

// ---------------------------------------------------------------- tiny scene engine
// A render() call lists the elements a step wants (keyed). Elements already on screen tween
// from where they are to their new attributes; new ones fade in; missing ones fade out.
const EASE = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const NUMRE = /-?\d*\.?\d+(?:e[-+]?\d+)?/gi;
function lerp(a, b, t, key) {
  if (a === undefined || a === null) return b;
  if (typeof a === 'number' && typeof b === 'number') return a + (b - a) * t;
  if (typeof a === 'string' && typeof b === 'string' && a !== b) {
    if (key === 'd') {
      if (a.replace(NUMRE, '#') !== b.replace(NUMRE, '#')) return b;
      const na = a.match(NUMRE), nb = b.match(NUMRE); let i = 0;
      return b.replace(NUMRE, () => { const v = +na[i] + (+nb[i] - +na[i]) * t; i++; return v.toFixed(1); });
    }
    if (isColor(a) && isColor(b)) return t >= 1 ? b : mix(a, b, t);
  }
  return b;
}

class Scene {
  constructor(host, tip) {
    this.host = host;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'lol-canvas';
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('class', 'lol-svg');
    host.append(this.canvas, this.svg);
    this.nodes = new Map(); this.raf = 0;
    const move = e => {
      const el = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
      if (el && el.__tip) { const r = host.getBoundingClientRect(); tip.show(e.clientX - r.left, e.clientY - r.top, el.__tip); }
      else tip.hide();
    };
    this.svg.addEventListener('pointermove', move);
    this.svg.addEventListener('pointerdown', move);
    this.svg.addEventListener('pointerleave', () => tip.hide());
  }
  render(specs, dur = 700) {
    const now = performance.now(), seen = new Set();
    for (const s of specs) {
      if (!s || seen.has(s.k)) continue;
      seen.add(s.k);
      let n = this.nodes.get(s.k);
      const to = Object.assign({ opacity: 1 }, s.a);
      if (!n || n.type !== s.t || n.dying) {
        if (n) n.el.remove();
        const el = document.createElementNS(NS, s.t);
        const from = s.from || {};
        n = { el, type: s.t, cur: Object.assign({}, to, from, { opacity: from.opacity != null ? from.opacity : 0 }) };
        this.nodes.set(s.k, n);
      }
      n.spec = s; n.from = Object.assign({}, n.cur); n.to = to;
      n.dying = false; n.done = false; n.dur = dur; n.t0 = now + (dur ? (s.delay || 0) : 0);
      const el = n.el, stat = s.s || {};
      for (const k of el.__s || []) if (!(k in stat)) el.removeAttribute(k);
      for (const k in stat) el.setAttribute(k, stat[k]);
      el.__s = Object.keys(stat);
      el.setAttribute('class', s.cls || '');
      if (s.t === 'text' && !s.f && s.text != null && el.__text !== s.text) { el.textContent = s.text; el.__text = s.text; }
      if (s.f) el.__text = null;
      el.__tip = s.tip || null;
      if (s.tip) el.setAttribute('data-tip', ''); else el.removeAttribute('data-tip');
      this.svg.appendChild(el);
    }
    for (const [k, n] of this.nodes) if (!seen.has(k) && !n.dying) {
      n.dying = true; n.done = false; n.from = Object.assign({}, n.cur);
      n.to = Object.assign({}, n.cur, { opacity: 0 }); n.t0 = now; n.dur = dur * .5;
      n.el.removeAttribute('data-tip'); n.el.__tip = null;
    }
    if (!dur) this.step(now + 1e9); else this.kick();
  }
  step(now) {
    let active = false;
    for (const [k, n] of this.nodes) {
      if (n.done) continue;
      const e = n.dur > 0 ? clamp((now - n.t0) / n.dur, 0, 1) : 1, t = EASE(e);
      for (const key in n.to) n.cur[key] = lerp(n.from[key], n.to[key], t, key);
      this.apply(n);
      if (e >= 1) { n.done = true; if (n.dying) { n.el.remove(); this.nodes.delete(k); } } else active = true;
    }
    return active;
  }
  apply(n) {
    const el = n.el, c = n.cur;
    for (const key in c) {
      if (key.charCodeAt(0) === 95) continue; // "__name": tweened parameter, not an attribute
      const v = c[key];
      el.setAttribute(key, typeof v === 'number' ? Math.round(v * 100) / 100 : v);
    }
    if (n.spec.f) {
      const o = n.spec.f(c);
      for (const key in o) {
        if (key === 'text') { if (el.textContent !== o.text) el.textContent = o.text; }
        else el.setAttribute(key, typeof o[key] === 'number' ? Math.round(o[key] * 100) / 100 : o[key]);
      }
    }
  }
  kick() {
    if (this.raf) return;
    const loop = now => { this.raf = 0; if (this.step(now)) this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }
}

// ---------------------------------------------------------------- drawing helpers
const fsz = (h, w, text) => {
  let f = clamp(h * .5, 8, 13);
  if (w && text != null) f = Math.min(f, (w - 6) / (Math.max(1, String(text).length) * .62));
  return Math.max(6.5, f);
};
function R(out, k, x, y, w, h, fill, o = {}) {
  out.push({ k, t: 'rect', a: { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: o.rx ?? 3, fill,
    stroke: o.stroke ?? 'none', 'stroke-width': o.sw ?? 1, opacity: o.op ?? 1 }, tip: o.tip, delay: o.delay, from: o.from });
}
function T(out, k, x, y, text, o = {}) {
  out.push({ k, t: 'text', a: Object.assign({ x, y, 'font-size': o.fs ?? 12, fill: o.color ?? C.ink, opacity: o.op ?? 1 }, o.a || {}),
    s: { 'text-anchor': o.anchor ?? 'start', 'dominant-baseline': 'central' }, text,
    cls: (o.mono === false ? '' : 'm') + (o.bold ? ' b' : '') + (o.tip ? ' hit' : ''),
    tip: o.tip, delay: o.delay, from: o.from, f: o.f });
}
function cell(out, k, x, y, w, h, text, o = {}) {
  const fill = o.fill ?? C.surface;
  R(out, k + '.r', x, y, w, h, fill, { stroke: o.stroke ?? C.cell_line, sw: o.sw, op: o.op, tip: o.tip, delay: o.delay, from: o.from, rx: o.rx });
  if (text == null && !o.f) return;
  const tx = o.anchor === 'start' ? x + 5 : o.anchor === 'end' ? x + w - 5 : x + w / 2;
  T(out, k + '.t', tx, y + h / 2, text, { fs: o.fs ?? fsz(h, w, text), color: o.color ?? textOn(fill), op: o.op,
    anchor: o.anchor ?? 'middle', bold: o.bold, delay: o.delay, mono: o.mono, from: o.fromT, f: o.f, a: o.ta });
}
function L(out, k, x1, y1, x2, y2, stroke, o = {}) {
  out.push({ k, t: 'line', a: { x1, y1, x2, y2, stroke, 'stroke-width': o.sw ?? 1, opacity: o.op ?? 1 },
    s: o.dash ? { 'stroke-dasharray': o.dash } : {}, delay: o.delay, from: o.from, tip: o.tip });
}
function P(out, k, d, o = {}) {
  out.push({ k, t: 'path', a: { d, fill: o.fill ?? 'none', stroke: o.stroke ?? C.flow, 'stroke-width': o.sw ?? 1.5, opacity: o.op ?? 1 },
    s: Object.assign({ 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, o.dash ? { 'stroke-dasharray': o.dash } : {}),
    delay: o.delay, from: o.from, tip: o.tip });
}
function arrow(x1, y1, x2, y2, dir = 'h', hs = 5, bend = .5) {
  let c1x, c1y, c2x, c2y;
  if (dir === 'h') { const dx = (x2 - x1) * bend; c1x = x1 + dx; c1y = y1; c2x = x2 - dx; c2y = y2; }
  else { const dy = (y2 - y1) * bend; c1x = x1; c1y = y1 + dy; c2x = x2; c2y = y2 - dy; }
  let ux = x2 - c2x, uy = y2 - c2y; const m = Math.hypot(ux, uy);
  if (m < 1e-6) { ux = dir === 'h' ? Math.sign(x2 - x1) || 1 : 0; uy = dir === 'h' ? 0 : Math.sign(y2 - y1) || 1; } else { ux /= m; uy /= m; }
  const ax = x2 - ux * hs, ay = y2 - uy * hs, px = -uy * hs * .6, py = ux * hs * .6;
  return `M${n1(x1)} ${n1(y1)}C${n1(c1x)} ${n1(c1y)} ${n1(c2x)} ${n1(c2y)} ${n1(x2)} ${n1(y2)}` +
    `M${n1(ax + px)} ${n1(ay + py)}L${n1(x2)} ${n1(y2)}L${n1(ax - px)} ${n1(ay - py)}`;
}
// split a total width into columns by weight
function columns(x, width, ws) {
  const s = ws.reduce((a, b) => a + b, 0); let cx = x;
  return ws.map(w => { const o = { x: cx, w: width * w / s }; cx += o.w; return o; });
}

// center a scene's content in the stage (shifts every position, including tweened "__" geometry)
const XK = ['x', 'x1', 'x2', 'cx', '__x0', '__x1'], YK = ['y', 'y1', 'y2', 'cy', '__y0', '__y1'];
function bbox(out) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  const add = (a, b, c, e) => { x0 = Math.min(x0, a, c); y0 = Math.min(y0, b, e); x1 = Math.max(x1, a, c); y1 = Math.max(y1, b, e); };
  for (const sp of out) {
    const a = sp.a;
    if (sp.t === 'rect') add(a.x, a.y, a.x + a.width, a.y + a.height);
    else if (sp.t === 'line' && a.x1 != null) add(a.x1, a.y1, a.x2, a.y2);
    else if (sp.t === 'text' && a.x != null) {
      const fs = a['font-size'] || 12, len = (sp.text ? String(sp.text).length : 0) * fs * .6, an = sp.s && sp.s['text-anchor'];
      const l = an === 'end' ? a.x - len : an === 'middle' ? a.x - len / 2 : a.x;
      add(l, a.y - fs / 2, l + len, a.y + fs / 2);
    } else if (sp.t === 'path') {
      const n = a.d.match(NUMRE).map(Number);
      for (let i = 0; i + 1 < n.length; i += 2) add(n[i], n[i + 1], n[i], n[i + 1]);
    }
    if (a.__x0 != null) add(a.__x0, a.__y1, a.__x1, a.__y0);
  }
  return { x0, y0, x1, y1 };
}
function shift(out, dx, dy) {
  for (const sp of out) for (const o of [sp.a, sp.from]) {
    if (!o) continue;
    for (const k of XK) if (typeof o[k] === 'number') o[k] += dx;
    for (const k of YK) if (typeof o[k] === 'number') o[k] += dy;
    if (typeof o.d === 'string') { let i = 0; o.d = o.d.replace(NUMRE, m => (+m + (i++ % 2 ? dy : dx)).toFixed(1)); }
  }
  return out;
}
const centered = (fn, vert = true) => d => {
  const out = fn(d), b = bbox(out);
  if (!(b.x1 > b.x0)) return out;
  const dx = Math.max(0, (d.W - (b.x1 - b.x0)) / 2 - b.x0);
  const dy = vert ? Math.max(0, Math.min((d.H - (b.y1 - b.y0)) / 2 - b.y0, 80)) : 0;
  return dx > 1 || dy > 1 ? shift(out, dx, dy) : out;
};

let S = null, RM = false;
function dims() {
  const W = S.host.clientWidth || 640, H = S.host.clientHeight || 420, narrow = W < 560;
  return { W, H, narrow, pad: narrow ? 10 : 22, fs: narrow ? 10 : 12 };
}

// ================================================================ chapter 1: the data
function funnel(out, d, level) {
  const items = [[cnt.rows, 'rows: players + teams', 'rows'], [cnt.team_rows, 'team rows, 2 per game', 'team rows'],
    [cnt.clean_rows, 'team rows with minute-25 data', 'with 25-min data']];
  const bh = d.narrow ? 11 : 14, gap = 5, maxW = (d.W - 2 * d.pad) * (d.narrow ? .34 : .3);
  items.forEach(([n, lab, sl], i) => {
    if (i > level) return;
    const y = d.pad + i * (bh + gap), w = maxW * n / cnt.rows, on = i === level;
    const pw = i ? maxW * items[i - 1][0] / cnt.rows : 0;
    R(out, 'fb' + i, d.pad, y, w, bh, on ? C.accent : C.frame, { rx: 2, op: on ? 1 : .45, from: { width: pw }, tip: `<b>${int(n)}</b> ${lab}` });
    T(out, 'fl' + i, d.pad + w + 8, y + bh / 2, null, { fs: d.narrow ? 10 : 12, color: on ? C.ink : C.muted,
      a: { __n: n }, from: { __n: i ? items[i - 1][0] : n, x: d.pad + pw + 8 },
      f: q => ({ text: `${int(q.__n)} ${d.narrow ? sl : lab}` }) });
  });
  return d.pad + 3 * (bh + gap) + 6;
}

const RAWC = [
  { c: 'position', h: 'position', w: .95 }, { c: 'side', h: 'side', w: .7 }, { c: 'who', h: 'player / team', w: 1.45 },
  { c: 'champion', h: 'champion', w: 1.15, wide: 1 }, { c: 'result', h: 'result', w: .75 },
  { c: 'goldat25', h: 'goldat25', w: 1.05 }, { c: 'xpat25', h: 'xpat25', w: 1.05, wide: 1 }, { c: 'killsat25', h: 'killsat25', w: .95 }];
const RI = Object.fromEntries(GM.raw_cols.map((c, i) => [c, i]));
const RNUM = ['goldat25', 'xpat25', 'killsat25'];
const RMAX = Object.fromEntries(RNUM.map(c => [c, Math.max(...GM.raw.map(r => r[RI[c]] || 0))]));
const rawVal = (row, c) => c === 'who' ? (row[RI.playername] ?? row[RI.teamname]) : row[RI[c]];
const rowKey = i => i >= 10 ? 'T' + (i - 10) : 'R' + i;

function rawTable(out, d, y0, mode) {
  const cols = RAWC.filter(c => !(d.narrow && c.wide));
  const tw = Math.min(d.W - 2 * d.pad, 760);
  const xs = columns(d.pad, tw, cols.map(c => c.w)).map((g, i) => Object.assign({}, cols[i], g));
  const rows = mode === 0 ? GM.raw.map((r, i) => i) : [10, 11];
  const nr = rows.length + 1 + (mode === 2 ? 1.4 : 0);
  const rh = Math.min(mode === 0 ? 25 : 34, (d.H - y0 - d.pad - (mode === 0 ? 24 : 56)) / nr);
  xs.forEach(c => T(out, 'rh-' + c.c, c.x + c.w / 2, y0 + rh / 2, c.h, { fs: Math.min(11, fsz(rh, c.w, c.h)), color: C.muted, anchor: 'middle' }));
  rows.forEach((ri, j) => {
    const row = GM.raw[ri], y = y0 + (j + 1) * rh, team = ri >= 10, who = rawVal(row, 'who');
    xs.forEach(c => {
      const v = rawVal(row, c.c); let txt = v == null ? '–' : String(v), fill = C.surface;
      if (c.c === 'result') { txt = v ? 'win' : 'loss'; fill = mix(C.zero, v ? C.good : C.bad, .55); }
      if (RNUM.includes(c.c)) { txt = int(v); fill = mix(C.zero, C.flow, .06 + .6 * v / RMAX[c.c]); }
      const colName = c.c === 'who' ? (team ? 'teamname' : 'playername') : c.c;
      let tip = `<b>${colName}</b> = ${txt}<br>${row[RI.position]} row · ${who}`;
      if (team && RNUM.includes(c.c)) {
        const sum = GM.raw.filter(r => r[RI.side] === row[RI.side] && r[RI.position] !== 'team').reduce((a, r) => a + r[RI[c.c]], 0);
        tip += `<br>= sum of its 5 players (${int(sum)})`;
      }
      cell(out, rowKey(ri) + '-' + c.c, c.x + 1, y + 1, c.w - 2, rh - 2, txt,
        { fill, stroke: team ? C.accent : C.cell_line, sw: team ? 1.5 : 1, tip, bold: team && c.c === 'who' });
    });
  });
  if (mode === 0) {
    T(out, 'cap', d.pad, y0 + (rows.length + 1) * rh + 12,
      `${GM.league} ${GM.date.slice(0, 10)} · game ${GM.game} · patch ${GM.patch} · ${int(GM.gamelength)} s`,
      { fs: d.narrow ? 9.5 : 11, color: C.muted });
  }
  if (mode === 2) {
    const de = GM.dropped_example, y = y0 + 3 * rh + 10;
    xs.forEach(c => {
      const nan = RNUM.includes(c.c);
      const txt = nan ? 'NaN' : { position: 'team', side: de.side, who: de.team, champion: '–', result: de.result ? 'win' : 'loss' }[c.c];
      cell(out, 'DR-' + c.c, c.x + 1, y + 1, c.w - 2, rh - 2, txt, { fill: nan ? C.empty : C.surface, color: nan ? C.muted : C.ink,
        stroke: C.bad, op: .8, delay: 100, tip: `${de.league} game ${de.gameid}<br>all 15 minute-25 columns are empty` });
    });
    const xe = xs[xs.length - 1].x + xs[xs.length - 1].w;
    L(out, 'DR-x', d.pad, y + rh / 2, xe, y + rh / 2, C.bad, { sw: 2, from: { x2: d.pad }, delay: 450 });
    T(out, 'DR-l', d.pad, y + rh + 14, `dropped: ${int(cnt.missing_rows)} rows like this (${pct(cnt.missing_frac)})`, { fs: d.narrow ? 10 : 12, color: C.bad, delay: 450 });
    const by = Object.entries(cnt.missing_by_league).map(([k, v]) => `${k} ${int(v)}`).join(' · ');
    T(out, 'DR-b', d.pad, y + rh + 32, `mostly ${by} · …`, { fs: d.narrow ? 9.5 : 11, color: C.muted, delay: 450 });
  }
}
function sceneRaw(mode) { return d => { const out = []; const y0 = funnel(out, d, mode); rawTable(out, d, y0, mode); return out; }; }

// minute-25 snapshot, one column per team
const SNAP = ['goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25', 'assistsat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25',
  'opp_killsat25', 'opp_deathsat25', 'opp_assistsat25', 'golddiffat25', 'xpdiffat25', 'csdiffat25'];
const SPREAD = c => /gold|xp/.test(c) ? 3000 : /cs/.test(c) ? 40 : 6;
function sceneSnap(d) {
  const out = [], top = d.pad, hh = d.narrow ? 36 : 44, gut = d.narrow ? 20 : 34;
  const tw = Math.min(d.W - 2 * d.pad, 560), lw = tw * (d.narrow ? .37 : .36), vw = (tw - lw - gut) / 2;
  const rh = Math.min(30, (d.H - top - hh - d.pad - 6) / SNAP.length);
  const X = i => d.pad + lw + i * (vw + gut), Y = j => top + hh + 4 + j * rh;
  TEAMS.forEach((t, i) => {
    cell(out, `T${i}-who`, X(i), top, vw, hh * .52, t.team, { stroke: C.accent, sw: 1.5, bold: true, tip: `${t.team}<br>${t.side} side · ${t.result ? 'won' : 'lost'} the game` });
    cell(out, `T${i}-result`, X(i) + vw / 2 - 25, top + hh * .6, 50, hh * .36, t.result ? 'won' : 'lost', { fill: mix(C.zero, t.result ? C.good : C.bad, .6) });
  });
  SNAP.forEach((c, j) => {
    T(out, 'sn-' + c, X(0) - 6, Y(j) + rh / 2, c, { anchor: 'end', fs: Math.min(12, fsz(rh, lw, c)), color: C.muted, tip: `<b>${c}</b><br>${DESC[c]}` });
    TEAMS.forEach((t, i) => {
      const v = t.at25[c], o = TEAMS[1 - i].at25[c], diff = /diff/.test(c);
      const rel = diff ? v / SPREAD(c) : (v - o) / SPREAD(c);
      cell(out, `T${i}-${c}`, X(i), Y(j) + 1, vw, rh - 2, int(v), { fill: div(rel, 1),
        tip: `<b>${t.team}</b> · ${c} = ${int(v)}<br>${DESC[c]}${diff ? '' : `<br>${TEAMS[1 - i].team}: ${int(o)}`}` });
    });
  });
  ['gold', 'xp', 'cs'].forEach((s, m) => {
    const ja = SNAP.indexOf(s + 'at25'), jb = SNAP.indexOf('opp_' + s + 'at25');
    const x1 = X(0) + vw, x2 = X(1), y1 = Y(ja) + rh / 2, y2 = Y(jb) + rh / 2;
    P(out, 'mir' + m, arrow(x1, y1, x2, y2, 'h', 4, .9), { stroke: C.flow, sw: 1.3, op: .85, delay: 300 + m * 120,
      tip: `${TEAMS[0].team}'s ${s}at25 is ${TEAMS[1].team}'s opp_${s}at25` });
  });
  return out;
}

function sceneHist(d) {
  const out = [], e = M.hist.edges, nb = e.length - 1, max = Math.max(...M.hist.win, ...M.hist.loss);
  const x0 = d.pad + 4, x1 = d.W - d.pad - 4, top = d.pad + 30, bot = d.H - d.pad - 28, mid = (top + bot) / 2, half = (bot - top) / 2 - 2;
  const X = v => x0 + (v - e[0]) / (e[nb] - e[0]) * (x1 - x0), bw = (x1 - x0) / nb;
  const lo = i => i === 0 ? `≤ ${int(e[1])}` : int(e[i]), hi = i => i === nb - 1 ? 'more' : int(e[i + 1]);
  for (let i = 0; i < nb; i++) {
    const hw = M.hist.win[i] / max * half, hl = M.hist.loss[i] / max * half;
    R(out, 'hw' + i, X(e[i]) + .5, mid - hw, bw - 1, hw, C.good, { rx: 1, from: { y: mid, height: 0 }, delay: i * 10,
      tip: `gold lead ${lo(i)} to ${hi(i)}<br><b>${int(M.hist.win[i])}</b> rows won` });
    R(out, 'hl' + i, X(e[i]) + .5, mid, bw - 1, hl, C.bad, { rx: 1, from: { height: 0 }, delay: i * 10,
      tip: `gold lead ${lo(i)} to ${hi(i)}<br><b>${int(M.hist.loss[i])}</b> rows lost` });
  }
  L(out, 'hax', x0, mid, x1, mid, C.frame);
  for (const v of [-20000, -10000, 0, 10000, 20000])
    T(out, 'ht' + v, X(v), bot + 14, v === 0 ? '0' : (v > 0 ? '+' : MINUS) + Math.abs(v / 1000) + 'k', { anchor: 'middle', fs: 10.5, color: C.muted });
  T(out, 'hxl', (x0 + x1) / 2, d.pad + 8, 'gold lead at minute 25 (golddiffat25)', { anchor: 'middle', fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
  T(out, 'hwl', x0 + 2, top + 8, '▲ won', { fs: 12, color: C.good, bold: true, mono: false });
  T(out, 'hll', x0 + 2, bot - 8, '▼ lost', { fs: 12, color: C.bad, bold: true, mono: false });
  const g = Math.abs(T1.at25.golddiffat25);
  L(out, 'hm0', X(-g), top - 4, X(-g), bot, C.accent, { dash: '3 3', sw: 1.2 });
  L(out, 'hm1', X(g), top - 4, X(g), bot, C.accent, { dash: '3 3', sw: 1.2 });
  T(out, 'hml', X(g) + 6, top + 26, `our game: ±${int(g)}`, { fs: d.narrow ? 10 : 11.5, color: C.accent, bold: true });
  T(out, 'hmr', x1 - 2, top + 8, `ahead in gold → won ${pct(cnt.ahead_win_rate)}`, { anchor: 'end', fs: d.narrow ? 10 : 11.5, color: C.ink, mono: false,
    tip: `${int(cnt.ahead_rows)} rows had a gold lead above 0 at minute 25; ${pct(cnt.ahead_win_rate)} of them won` });
  return out;
}

// ================================================================ chapter 2: features
const CAND = [...SNAP, 'side', 'gamelength', 'league'];
const USED = new Set(['goldat25', 'xpat25', 'csat25', 'killsat25', 'deathsat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'side', 'gamelength', 'league']);
const nameKey = c => 'sn-' + (c === 'side_binary' ? 'side' : c);
const xKey = c => 'T1-' + (c === 'side_binary' ? 'side' : c);
function sceneInputs(d) {
  const out = [], top = d.pad + 22;
  const rh = Math.min(28, (d.H - top - d.pad) / CAND.length);
  const tw = Math.min(d.W - 2 * d.pad, 560), lw = tw * .36, vw = tw * .24, x0 = d.pad;
  T(out, 'in-h', x0, d.pad + 7, 'T1 row: 18 columns → 11 model inputs', { fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
  CAND.forEach((c, j) => {
    const y = top + j * rh, used = USED.has(c);
    T(out, 'sn-' + c, x0 + lw - 6, y + rh / 2, c, { anchor: 'end', fs: Math.min(12, fsz(rh, lw, c)), color: used ? C.ink : C.muted,
      op: used ? 1 : .55, tip: `<b>${c}</b><br>${DESC[c]}` });
    const txt = c === 'side' ? 'Red → 1' : c === 'gamelength' ? int(T1.gamelength) + ' s' : c === 'league' ? T1.league : int(T1.at25[c]);
    cell(out, 'T1-' + c, x0 + lw, y + 1, vw - 4, rh - 2, txt, { fill: used ? mix(C.zero, C.accent, .2) : C.surface,
      stroke: used ? C.accent : C.cell_line, op: used ? 1 : .4, tip: `<b>T1</b> · ${c} = ${txt}<br>${DESC[c]}` });
    const note = !used ? 'not used' : c === 'gamelength' ? (d.narrow ? 'unknown at 25:00!' : 'input, but unknown at 25:00!') : 'input';
    T(out, 'st-' + c, x0 + lw + vw + 2, y + rh / 2, note, { fs: Math.min(11.5, rh * .48), mono: false,
      color: !used ? C.muted : c === 'gamelength' ? C.bad : C.accent, op: used ? 1 : .7 });
    if (!used) L(out, 'sx-' + c, x0 + lw + 4, y + rh / 2, x0 + lw + vw - 8, y + rh / 2, C.muted, { from: { x2: x0 + lw + 4 }, delay: 250 + j * 25 });
  });
  return out;
}

function sceneSplit(d) {
  const out = [], x0 = d.pad, w = d.W - 2 * d.pad, fs = d.narrow ? 10.5 : 12;
  const code = 'train_test_split(X, y, test_size=0.3, random_state=103)';
  T(out, 'sp-code', x0, d.pad + 8, code, { fs: Math.min(12, (w - 4) / (code.length * .62)), color: C.muted });
  const yc = d.pad + 28, bh = d.narrow ? 44 : 58, y = yc + 48;
  const wt = w * met.n_train / cnt.clean_rows;
  R(out, 'sp-tr', x0, y, wt - 3, bh, mix(C.zero, C.flow, .6), { from: { width: 0 }, tip: `<b>${int(met.n_train)}</b> training rows: the model learns from these` });
  R(out, 'sp-te', x0 + wt, y, w - wt, bh, mix(C.zero, C.accent, .6), { from: { x: x0 + w, width: 0 }, delay: 150,
    tip: `<b>${int(met.n_test)}</b> test rows: hidden until the end, used only for grading` });
  const ftr = mix(C.zero, C.flow, .6), fte = mix(C.zero, C.accent, .6);
  T(out, 'sp-trl', x0 + 10, y + bh * .34, 'training', { fs, color: textOn(ftr), bold: true, mono: false });
  T(out, 'sp-trn', x0 + 10, y + bh * .7, `${int(met.n_train)} rows · 70%`, { fs: fs - .5, color: textOn(ftr) });
  T(out, 'sp-tel', x0 + wt + 8, y + bh * .34, 'test', { fs, color: textOn(fte), bold: true, mono: false });
  T(out, 'sp-ten', x0 + wt + 8, y + bh * .7, `${int(met.n_test)} · 30%`, { fs: fs - .5, color: textOn(fte) });
  TEAMS.forEach((t, i) => {
    const cx = x0 + wt * (.42 + i * .26), cw = d.narrow ? 54 : 70;
    cell(out, 'chip' + i, cx - cw / 2, yc, cw, 22, SHORT[i] + ' row', { stroke: C.accent, sw: 1.5,
      tip: `${t.team}'s row from our game: ${t.in_test ? 'test' : 'training'} set` });
    P(out, 'chipa' + i, arrow(cx, yc + 24, cx, y - 3, 'v', 5), { stroke: C.flow, sw: 1.4, delay: 300 });
  });
  // what row-level splitting does to games
  const y2 = y + bh + 44, gb = d.narrow ? 16 : 20, g = cnt.games_clean;
  T(out, 'sp-gh', x0, y2 - 14, `the ${int(g)} games, by where their 2 rows went:`, { fs: fs - .5, color: C.muted, mono: false });
  const segs = [[cnt.games_both_train, 'both in training', C.flow], [cnt.games_split, 'one in each', C.frame], [cnt.games_both_test, 'both in test', C.accent]];
  let gx = x0;
  segs.forEach(([n, lab, col], i) => {
    const sw = w * n / g;
    R(out, 'sp-g' + i, gx, y2, sw - 2, gb, mix(C.zero, col, .55), { from: { width: 0 }, delay: 400 + i * 120, tip: `<b>${int(n)}</b> games: ${lab}` });
    T(out, 'sp-gl' + i, gx + (i === 2 ? sw - 2 : 0), y2 + gb + 12, `${int(n)} ${d.narrow && i !== 1 ? lab.replace('both in ', '') : lab}`,
      { fs: d.narrow ? 9.5 : 11, color: C.muted, anchor: i === 2 ? 'end' : 'start', delay: 400 + i * 120 });
    gx += sw;
  });
  return out;
}

const fmtStat = (c, v) => /gold|xp|gamelength/.test(c) ? int(v) : c === 'side_binary' ? fx(v, 3) : /cs/.test(c) ? fx(v, 1) : fx(v, 2);
function sceneScale(d) {
  const out = [], top = d.pad + 44;
  const rh = Math.min(32, (d.H - top - d.pad) / (NUM.length + 1));
  const tw = Math.min(d.W - 2 * d.pad, 620);
  const [cn, cx, cm, cs, cz] = columns(d.pad, tw, [1.55, 1, 1, .9, .9]);
  T(out, 'f-1', d.pad, d.pad + 9, 'z = (x − mean) / std', { fs: d.narrow ? 13 : 15, bold: true });
  T(out, 'f-2', d.pad, d.pad + 29, `mean and std of the ${int(met.n_train)} training rows`, { fs: d.narrow ? 10 : 11.5, color: C.muted, mono: false });
  [['x (T1)', cx], ['mean', cm], ['std', cs], ['z', cz]].forEach(([h, c], i) =>
    T(out, 'sch' + i, c.x + c.w / 2, top + rh / 2, h, { anchor: 'middle', fs: Math.min(11.5, rh * .48), color: C.muted }));
  NUM.forEach((c, j) => {
    const i = IDX[c], y = top + (j + 1) * rh, x = T1.x[i], z = T1.z[i];
    const xs = c === 'side_binary' ? '1' : int(x);
    T(out, nameKey(c), cn.x + cn.w - 6, y + rh / 2, c, { anchor: 'end', fs: Math.min(12, fsz(rh, cn.w, c)), color: C.ink, tip: `<b>${c}</b><br>${DESC[c]}` });
    cell(out, xKey(c), cx.x + 1, y + 1, cx.w - 2, rh - 2, xs, { fill: mix(C.zero, C.accent, .2), stroke: C.accent, tip: `T1's ${c} = ${xs}` });
    cell(out, 'mu-' + c, cm.x + 1, y + 1, cm.w - 2, rh - 2, fmtStat(c, M.mean[i]), { tip: `average ${c} over training rows: ${M.mean[i]}` });
    cell(out, 'sd-' + c, cs.x + 1, y + 1, cs.w - 2, rh - 2, fmtStat(c, M.scale[i]), { tip: `standard deviation of ${c}: ${M.scale[i]}` });
    cell(out, 'z1-' + c, cz.x + 1, y + 1, cz.w - 2, rh - 2, fx(z, 2), { fill: div(z, 2.5), bold: true, delay: 200 + j * 50,
      tip: `(${xs} − ${M.mean[i]}) / ${M.scale[i]}<br>= <b>${fx(z, 4)}</b>` });
  });
  return out;
}

function zList(out, d, x0, lw, zw, top, rh) {
  NUM.forEach((c, j) => {
    const i = IDX[c], y = top + j * rh, z = T1.z[i];
    T(out, nameKey(c), x0 + lw - 6, y + rh / 2, c, { anchor: 'end', fs: Math.min(12, fsz(rh, lw, c)), tip: `<b>${c}</b><br>${DESC[c]}` });
    cell(out, 'z1-' + c, x0 + lw, y + 1, zw, rh - 2, fx(z, 2), { fill: div(z, 2.5), bold: true, tip: `scaled ${c} = <b>${fx(z, 4)}</b>` });
  });
}
function sceneOneHot(d) {
  const out = [], top = d.pad + 26, w = d.W - 2 * d.pad;
  const left = Math.min(d.narrow ? w * .52 : 300, w * .5), lw = left * .64, zw = left * .32;
  const rh = Math.min(26, (d.H - top - d.pad - 24) / NUM.length);
  T(out, 'oh-h1', d.pad, d.pad + 9, '10 scaled numbers', { fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
  zList(out, d, d.pad, lw, zw, top, rh);
  const xr = d.pad + left + (d.narrow ? 10 : 30), rw = d.W - d.pad - xr;
  T(out, 'oh-h2', xr, d.pad + 9, '+ 47 league switches', { fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
  cell(out, 'T1-league', xr, top, Math.min(rw, 150), 24, 'league = ' + T1.league, { fill: mix(C.zero, C.accent, .2), stroke: C.accent,
    tip: `T1's league: ${T1.league} (the World Championship)` });
  const cols = 6, cs = Math.min(rw / cols, (d.H - top - 70 - d.pad) / Math.ceil(LEAGUES.length / cols), 40);
  const gy = top + 46;
  P(out, 'oh-a', arrow(xr + 20, top + 26, xr + 20, gy - 3, 'v', 5), { stroke: C.flow, sw: 1.4 });
  LEAGUES.forEach((lg, i) => {
    const on = i === LIDX, x = xr + (i % cols) * cs, y = gy + Math.floor(i / cols) * cs;
    cell(out, 'lg' + i, x + 1, y + 1, cs - 2, cs - 2, cs >= 16 ? (on ? '1' : '0') : null, {
      fill: on ? C.accent : C.zero, stroke: on ? C.accent : C.cell_line, delay: i * 12, rx: 2,
      color: on ? '#fff' : C.muted, tip: `league_${lg} = ${on ? 1 : 0}` });
  });
  const by = gy + Math.ceil(LEAGUES.length / cols) * cs + 14;
  T(out, 'oh-on', xr, by, `only league_${T1.league} is 1`, { fs: d.narrow ? 10 : 11.5, color: C.accent });
  T(out, 'oh-tot', d.pad, d.H - d.pad - 6, '10 + 47 = 57 numbers per row', { fs: d.narrow ? 11 : 13, bold: true });
  return out;
}

// ================================================================ chapter 3: the forward pass
function fwdGeom(d) {
  const top = d.pad + 26, n = NUM.length + 1;
  const rh = Math.min(32, (d.H - top - d.pad - 4) / (n + 1));
  const sumW = d.narrow ? 88 : 150, gapS = d.narrow ? 16 : 40;
  const tw = Math.min(d.W - 2 * d.pad - sumW - gapS, 560);
  const [cn, cz, c1, cw, c2, cp] = columns(d.pad, tw, [1.95, 1, .3, 1.05, .3, 1.1]);
  return { top, rh, cn, cz, c1, cw, c2, cp, sumX: d.pad + tw + gapS, sumW };
}
function sceneFwd(withSum) {
  return d => {
    const out = [], g = fwdGeom(d), fs = Math.min(11.5, g.rh * .48);
    [['z', g.cz], ['weight w', g.cw], ['w × z', g.cp]].forEach(([h, c], i) =>
      T(out, 'fh' + i, c.x + c.w / 2, g.top + g.rh / 2, h, { anchor: 'middle', fs, color: C.muted }));
    T(out, 'fh-t', d.pad, d.pad + 8, 'T1 row', { fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
    const rows = [...NUM, 'league'];
    rows.forEach((c, j) => {
      const y = g.top + (j + 1) * g.rh, mid = y + g.rh / 2;
      if (c === 'league') {
        T(out, 'sn-league', g.cn.x + g.cn.w - 6, mid, 'league ×47', { anchor: 'end', fs: Math.min(12, fsz(g.rh, g.cn.w, 'league ×47')), op: .6,
          tip: '47 one-hot league switches' });
        R(out, 'zL', g.cz.x + 1, y + 1, g.cz.w - 2, g.rh - 2, C.surface, { stroke: C.cell_line, op: .6 });
        const sw = (g.cz.w - 8) / LEAGUES.length;
        LEAGUES.forEach((lg, i) => R(out, 'lg' + i + '.r', g.cz.x + 4 + i * sw, y + 4, Math.max(.6, sw - .3), g.rh - 8,
          i === LIDX ? C.accent : C.empty, { rx: 0, stroke: 'none', tip: `league_${lg} = ${i === LIDX ? 1 : 0}` }));
        cell(out, 'wL', g.cw.x + 1, y + 1, g.cw.w - 2, g.rh - 2, 'all 0', { op: .6, tip: `all 47 league weights are exactly 0` });
        T(out, 'ox-league', g.c1.x + g.c1.w / 2, mid, '×', { anchor: 'middle', fs, color: C.muted, op: .6 });
        T(out, 'oq-league', g.c2.x + g.c2.w / 2, mid, '=', { anchor: 'middle', fs, color: C.muted, op: .6 });
        cell(out, 'pL', g.cp.x + 1, y + 1, g.cp.w - 2, g.rh - 2, '0', { op: .6, delay: 600, tip: 'league contributes 0' });
        return;
      }
      const i = IDX[c], z = T1.z[i], w = M.coef[i], p = T1.prod[i], op = w === 0 ? .55 : 1;
      T(out, nameKey(c), g.cn.x + g.cn.w - 6, mid, c, { anchor: 'end', fs: Math.min(12, fsz(g.rh, g.cn.w, c)), op, tip: `<b>${c}</b><br>${DESC[c]}` });
      cell(out, 'z1-' + c, g.cz.x + 1, y + 1, g.cz.w - 2, g.rh - 2, fx(z, 2), { fill: div(z, 2.5), op, tip: `scaled ${c} = ${fx(z, 4)}` });
      T(out, 'ox-' + c, g.c1.x + g.c1.w / 2, mid, '×', { anchor: 'middle', fs, color: C.muted, op });
      cell(out, 'w-' + c, g.cw.x + 1, y + 1, g.cw.w - 2, g.rh - 2, fx(w, 3), { fill: div(w, 1.1), op, delay: j * 40,
        tip: `weight for ${c}: <b>${fx(w, 6)}</b><br>learned in training${w === 0 ? '; LASSO set it to exactly 0' : ''}` });
      T(out, 'oq-' + c, g.c2.x + g.c2.w / 2, mid, '=', { anchor: 'middle', fs, color: C.muted, op });
      cell(out, 'p1-' + c, g.cp.x + 1, y + 1, g.cp.w - 2, g.rh - 2, fx(p, 3), { fill: div(p, .6), bold: true, op, delay: 250 + j * 70,
        tip: `${fx(z, 4)} × ${fx(w, 4)} = <b>${fx(p, 4)}</b><br>${p > 0 ? 'pushes toward a win' : p < 0 ? 'pushes toward a loss' : 'no effect'}` });
    });
    if (!withSum) R(out, 'spacer', g.sumX, g.top, g.sumW, 4, 'none', { op: 0 });
    if (withSum) {
      const n = rows.length, sx = g.sumX, bh = Math.max(24, Math.min(34, g.rh * 1.25));
      const sy = g.top + g.rh * (1 + n / 2) - bh * 1.5;
      const sum = T1.prod.reduce((a, b) => a + b, 0);
      rows.forEach((c, j) => {
        const y = g.top + (j + 1) * g.rh + g.rh / 2;
        P(out, 'ar-' + c, arrow(g.cp.x + g.cp.w, y, sx - 2, sy + bh / 2, 'h', 4, .55), { stroke: C.flow, sw: 1.1, op: c === 'league' || c === 'gamelength' ? .3 : .7, delay: j * 50 });
      });
      cell(out, 'sum', sx, sy, g.sumW, bh, `Σ = ${fx(sum, 3)}`, { tip: `sum of the 57 products: ${fx(sum, 4)}`, delay: 500 });
      cell(out, 'icpt', sx, sy + bh + 4, g.sumW, bh, `+ ${d.narrow ? 'b' : 'intercept'} ${fx(M.intercept, 0)}`,
        { tip: `intercept b = ${M.intercept}<br>the L1 penalty pushed it to exactly 0`, delay: 650, color: C.muted });
      cell(out, 'logit', sx, sy + 2 * (bh + 4), g.sumW, bh, `logit ${fx(T1.logit, 3)}`, { stroke: C.accent, sw: 2, bold: true, delay: 800,
        fill: div(T1.logit, 3), tip: `logit z = ${fx(T1.logit, 4)}<br>the model's raw score for T1` });
    }
    return out;
  };
}

// sigmoid plot
const ZMIN = -6, ZMAX = 6;
function sigPlot(out, g, d, o = {}) {
  const X = z => g.x0 + (z - ZMIN) / (ZMAX - ZMIN) * (g.x1 - g.x0), Y = p => g.y0 + p * (g.y1 - g.y0);
  const N = 60; let dd = '', flat = '';
  for (let i = 0; i <= N; i++) {
    const z = ZMIN + (ZMAX - ZMIN) * i / N;
    dd += (i ? 'L' : 'M') + n1(X(z)) + ' ' + n1(Y(sigmoid(z)));
    flat += (i ? 'L' : 'M') + n1(X(z)) + ' ' + n1(Y(.5));
  }
  const fs = d.narrow ? 9.5 : 11;
  L(out, 'sg-ax', g.x0, g.y0, g.x1, g.y0, C.frame);
  L(out, 'sg-ay', g.x0, g.y0, g.x0, g.y1, C.frame);
  L(out, 'sg-th', g.x0, Y(.5), g.x1, Y(.5), C.muted, { dash: '4 4', op: .7 });
  L(out, 'sg-z0', X(0), g.y0, X(0), g.y1, C.line, { op: .9 });
  for (const z of [-6, -4, -2, 0, 2, 4, 6]) T(out, 'sg-tz' + z, X(z), g.y0 + 12, z < 0 ? MINUS + -z : String(z), { anchor: 'middle', fs, color: C.muted });
  for (const p of [0, .5, 1]) T(out, 'sg-tp' + p, g.x0 - 5, Y(p), String(p), { anchor: 'end', fs, color: C.muted });
  T(out, 'sg-lx', g.x1, g.y0 - 9, 'logit z', { anchor: 'end', fs, color: C.muted });
  T(out, 'sg-ly', g.x0 + 6, g.y1 + 2, 'p(win)', { fs, color: C.muted });
  if (!o.noFormula) T(out, 'sg-f', g.x0 + 6, g.y1 + 20, 'σ(z) = 1 / (1 + e^(−z))', { fs: fs + 1, color: C.ink });
  T(out, 'sg-w', g.x1 - 4, Y(.5) - 9, 'predict win ↑', { anchor: 'end', fs, color: C.good, mono: false });
  T(out, 'sg-l', g.x1 - 4, Y(.5) + 10, 'predict loss ↓', { anchor: 'end', fs, color: C.bad, mono: false });
  P(out, 'sg-curve', dd, { stroke: C.ink, sw: 2, from: { d: flat } });
}
const geomAttrs = g => ({ __x0: g.x0, __x1: g.x1, __y0: g.y0, __y1: g.y1 });
const ptXY = q => {
  const z = clamp(q.__z, ZMIN, ZMAX);
  return [q.__x0 + (z - ZMIN) / (ZMAX - ZMIN) * (q.__x1 - q.__x0), q.__y0 + sigmoid(q.__z) * (q.__y1 - q.__y0)];
};
function sigPoint(out, k, g, z, color, name, o = {}) {
  const ga = geomAttrs(g);
  out.push({ k: k + 'gv', t: 'line', a: Object.assign({ __z: z, __r: 1, stroke: color, 'stroke-width': 1.3, opacity: .9 }, ga),
    from: o.reveal ? { __r: 0 } : undefined, s: { 'stroke-dasharray': '3 3' },
    f: q => { const [x, y] = ptXY(q), r = clamp(q.__r * 2, 0, 1); return { x1: x, y1: q.__y0, x2: x, y2: q.__y0 + (y - q.__y0) * r }; } });
  out.push({ k: k + 'gh', t: 'line', a: Object.assign({ __z: z, __r: 1, stroke: color, 'stroke-width': 1.3, opacity: .9 }, ga),
    from: o.reveal ? { __r: 0 } : undefined, s: { 'stroke-dasharray': '3 3' },
    f: q => { const [x, y] = ptXY(q), r = clamp(q.__r * 2 - 1, 0, 1); return { x1: x, y1: y, x2: x + (q.__x0 - x) * r, y2: y }; } });
  out.push({ k, t: 'circle', a: Object.assign({ __z: z, r: o.r ?? 6.5, fill: color, stroke: C.surface, 'stroke-width': 2 }, ga),
    from: o.reveal ? { r: 0 } : undefined, delay: o.reveal ? 500 : 0,
    tip: `${name}: logit ${fx(z, 3)} → σ = <b>${pct(sigmoid(z), 1)}</b>`,
    f: q => { const [x, y] = ptXY(q); return { cx: x, cy: y }; } });
  out.push({ k: k + 'l', t: 'text', a: Object.assign({ __z: z, fill: color, 'font-size': o.fs ?? 12 }, ga),
    s: { 'text-anchor': o.left ? 'end' : 'start', 'dominant-baseline': 'central' }, cls: 'm b', delay: o.reveal ? 700 : 0,
    f: q => { const [x, y] = ptXY(q); return { x: x + (o.left ? -11 : 11), y: y + (o.dy ?? -12), text: `${name} ${pct(sigmoid(q.__z), 1)}`.trim() }; } });
}

function sceneSig(d) {
  const out = [];
  cell(out, 'logit', d.pad, d.pad, d.narrow ? 120 : 150, 28, `logit ${fx(T1.logit, 3)}`, { stroke: C.accent, sw: 2, bold: true,
    fill: div(T1.logit, 3), tip: `T1's logit from the previous step: ${fx(T1.logit, 4)}` });
  const g = { x0: d.pad + 26, x1: d.W - d.pad - 6, y1: d.pad + 46, y0: d.H - d.pad - 22 };
  sigPlot(out, g, d);
  sigPoint(out, 'pt1', g, T1.logit, C.accent, 'T1', { reveal: true });
  const res = T1.result ? 'T1 won ✓' : 'T1 lost ✗';
  T(out, 'sg-res', g.x1 - 4, g.y1 + 24, `actual: ${res}`, { anchor: 'end', fs: d.narrow ? 11 : 13, color: C.good, bold: true, mono: false, delay: 900 });
  return out;
}

function contrib(out, d, box, teams, o = {}) {
  // teams: [{key, prods (10, by NUM order), color, label}]
  const rows = FEAT9, rh = Math.min(30, (box.h - 22) / rows.length);
  const lw = Math.min(box.w * .42, d.narrow ? 100 : 140);
  const bx0 = box.x + lw + 4, bx1 = box.x + box.w - 4, zx = (bx0 + bx1) / 2, sc = (bx1 - bx0) / 2 / (o.max ?? .65);
  const fs = d.narrow ? 9.5 : 11;
  T(out, 'cb-hl', bx0, box.y + 8, '← loss', { fs, color: C.muted, mono: false });
  T(out, 'cb-hr', bx1, box.y + 8, 'win →', { fs, color: C.muted, anchor: 'end', mono: false });
  L(out, 'cb-zero', zx, box.y + 18, zx, box.y + 20 + rows.length * rh, C.frame);
  const bh = (rh - 6) / teams.length;
  rows.forEach((c, j) => {
    const y = box.y + 20 + j * rh, i = IDX[c];
    T(out, 'cbn-' + c, bx0 - 6, y + rh / 2, c, { anchor: 'end', fs: Math.min(11.5, fsz(rh, lw, c)), color: C.ink, tip: `<b>${c}</b><br>${DESC[c]}` });
    teams.forEach((t, m) => {
      const v = t.prods[i], w = Math.min(Math.abs(v) * sc, (bx1 - bx0) / 2);
      R(out, t.key + '-' + c, v < 0 ? zx - w : zx, y + 3 + m * bh, w, bh - 1, t.sign ? (v < 0 ? C.neg : C.pos) : t.color, { rx: 1.5,
        from: { x: zx, width: 0 }, tip: `${t.label}: ${c} adds <b>${sg(v, 3)}</b> to the logit` });
    });
  });
  return box.y + 20 + rows.length * rh;
}

function sceneBoth(d) {
  const out = [], wide = !d.narrow;
  const cb = wide ? { x: d.pad, y: d.pad + 18, w: (d.W - 2 * d.pad) * .52, h: d.H - 2 * d.pad - 70 }
    : { x: d.pad, y: d.pad + 14, w: d.W - 2 * d.pad, h: (d.H - 2 * d.pad) * .52 };
  const teams = TEAMS.map((t, i) => ({ key: 'cb' + i, prods: t.prod, color: i ? C.accent : C.flow, label: t.team }));
  const yb = contrib(out, d, cb, teams);
  T(out, 'leg0', d.pad, d.pad + 4, '■ BLG', { fs: 11, color: C.flow, bold: true });
  T(out, 'leg1', d.pad + 52, d.pad + 4, '■ T1', { fs: 11, color: C.accent, bold: true });
  const fs = d.narrow ? 10.5 : 12;
  const tot = TEAMS[0].p + TEAMS[1].p;
  if (wide) {
    TEAMS.forEach((t, i) => T(out, 'bs' + i, d.pad, yb + 18 + i * 18, `${t.team}: logit ${fx(t.logit, 3)} → ${pct(t.p)} (${t.result ? 'won' : 'lost'})`,
      { fs, color: i ? C.accent : C.flow, bold: true }));
    T(out, 'bs2', d.pad, yb + 54, `together: ${pct(tot)}, not exactly 100%`, { fs, color: C.muted });
  }
  const g = wide ? { x0: d.pad + cb.w + 50, x1: d.W - d.pad - 6, y1: d.pad + 30, y0: d.H - d.pad - 24 }
    : { x0: d.pad + 24, x1: d.W - d.pad - 4, y1: cb.y + cb.h + 18, y0: d.H - d.pad - 18 };
  sigPlot(out, g, d, { noFormula: true });
  sigPoint(out, 'pt0', g, TEAMS[0].logit, C.flow, 'BLG', { left: true, dy: 12 });
  sigPoint(out, 'pt1', g, TEAMS[1].logit, C.accent, 'T1');
  if (!wide) T(out, 'bs2', g.x0 + 6, g.y1 + 22, `sum ${pct(tot)}`, { fs: 10, color: C.muted });
  return out;
}

// ================================================================ chapter 4: training
function coefBars(out, d, box, w, b, o = {}) {
  const rows = [...ORDER, 'league', 'intercept'], rh = Math.min(30, box.h / rows.length);
  const lw = d.narrow ? 92 : 128, vw = d.narrow ? 46 : 60;
  const bx0 = box.x + lw, bx1 = box.x + box.w - vw, zx = (bx0 + bx1) / 2, sc = (bx1 - bx0) / 2 / 1.15;
  const fs = Math.max(7.5, Math.min(11.5, rh * .5));
  L(out, 'cz', zx, box.y - 4, zx, box.y + rows.length * rh, C.frame);
  rows.forEach((c, j) => {
    const y = box.y + j * rh, mid = y + rh / 2;
    if (c === 'league') {
      T(out, 'cn-league', bx0 - 6, mid, 'league ×47', { anchor: 'end', fs, color: C.muted, tip: '47 one-hot league switches' });
      const sw = (bx1 - bx0) / LEAGUES.length;
      LEAGUES.forEach((lg, i) => R(out, 'lg' + i + '.r', bx0 + i * sw, y + rh * .3, Math.max(.6, sw - 1), rh * .4, C.empty,
        { rx: 0, stroke: 'none', tip: `league_${lg}: weight ${fx(LEAGUE_W[i], 3)}` }));
      T(out, 'cv-league', bx1 + 6, mid, 'all 0', { fs, color: C.muted });
      return;
    }
    const v = c === 'intercept' ? b : w[IDX[c]], bw = Math.abs(v) * sc;
    T(out, 'cn-' + c, bx0 - 6, mid, c, { anchor: 'end', fs: Math.min(fs, fsz(rh, lw, c)), color: v === 0 ? C.muted : C.ink,
      tip: `<b>${c}</b><br>${DESC[c] || 'the bias term'}` });
    R(out, 'cb-' + c, v < 0 ? zx - bw : zx, y + rh * .2, bw, rh * .6, v < 0 ? C.neg : C.pos, { rx: 2,
      from: { x: zx, width: 0 }, tip: `${c}: weight <b>${fx(v, 4)}</b>` });
    T(out, 'cv-' + c, bx1 + 6, mid, fx(v, 3), { fs, color: v === 0 ? C.muted : C.ink });
  });
  if (o.axis !== false) {
    for (const t of [-1, 0, 1]) T(out, 'ct' + t, zx + t * sc, box.y + rows.length * rh + 10, t === 0 ? '0' : (t > 0 ? '+1' : MINUS + '1'),
      { anchor: 'middle', fs: 10, color: C.muted });
  }
}
function sceneCoef(d) {
  const out = [];
  T(out, 'co-h', d.pad, d.pad + 8, `fitted weights, sorted by size (${NONZERO} of 57 not zero)`, { fs: d.narrow ? 10.5 : 12, color: C.muted, mono: false });
  coefBars(out, d, { x: d.pad, y: d.pad + 30, w: Math.min(d.W - 2 * d.pad, 640), h: d.H - 2 * d.pad - 48 }, M.coef.slice(0, NUM.length), M.intercept);
  return out;
}

function scenePath(d) {
  const out = [], x0 = d.pad + 30, x1 = d.W - d.pad - (d.narrow ? 54 : 96);
  const aT = d.pad + 22, aB = d.pad + (d.H - 2 * d.pad) * .56, bT = aB + 34, bB = d.H - d.pad - 22;
  const lx = c => x0 + (Math.log10(c) + 4) / 8 * (x1 - x0);
  const ay = v => (aT + aB) / 2 - v / 1.2 * (aB - aT) / 2;
  const fs = d.narrow ? 9.5 : 11;
  T(out, 'pa-h', d.pad, d.pad + 6, d.narrow ? 'each line = one weight, refit at each C' : 'each line = one weight, refit at each penalty strength C', { fs: d.narrow ? 10 : 11.5, color: C.muted, mono: false });
  L(out, 'pa-0', x0, ay(0), x1, ay(0), C.frame);
  for (const t of [-1, 1]) {
    L(out, 'pa-g' + t, x0, ay(t), x1, ay(t), C.line, { dash: '2 4' });
    T(out, 'pa-gt' + t, x0 - 5, ay(t), t > 0 ? '+1' : MINUS + '1', { anchor: 'end', fs, color: C.muted });
  }
  const firstL = M.path.find(p => p.league_nonzero > 0);
  if (firstL) {
    L(out, 'pa-lg', lx(firstL.C), aT + 22, lx(firstL.C), aB, C.muted, { dash: '1 3' });
    T(out, 'pa-lgt', lx(firstL.C) + 4, aB - 6, d.narrow ? 'leagues ≠ 0 →' : 'league weights start to turn on →', { fs, color: C.muted, mono: false });
  }
  NUM.forEach(c => {
    const i = IDX[c], wf = M.coef[i];
    const dd = M.path.map((p, k) => (k ? 'L' : 'M') + n1(lx(p.C)) + ' ' + n1(ay(p.w[i]))).join('');
    const flat = M.path.map((p, k) => (k ? 'L' : 'M') + n1(lx(p.C)) + ' ' + n1(ay(0))).join('');
    P(out, 'pl-' + c, dd, { stroke: wf > 0 ? C.pos : wf < 0 ? C.neg : C.muted, sw: 1.8, op: wf === 0 ? .5 : .95, from: { d: flat }, tip: `<b>${c}</b><br>final weight ${fx(wf, 3)}` });
  });
  // labels at the right end for the biggest four
  const last = M.path[M.path.length - 1];
  ORDER.slice(0, 4).forEach(c => T(out, 'pll-' + c, x1 + 4, ay(last.w[IDX[c]]), c.replace('at25', ''), { fs, color: W_OF(c) > 0 ? C.pos : C.neg }));
  // CV accuracy panel
  const amin = .45, amax = .86, by = a => bB - (a - amin) / (amax - amin) * (bB - bT);
  L(out, 'cv-ax', x0, bB, x1, bB, C.frame);
  T(out, 'cv-h', x0, bT - 12, 'mean 5-fold CV accuracy for the 10 C values tried', { fs: d.narrow ? 9.5 : 11, color: C.muted, mono: false });
  for (const a of [.5, .8]) T(out, 'cv-t' + a, x0 - 5, by(a), pct(a, 0), { anchor: 'end', fs, color: C.muted });
  P(out, 'cv-line', M.Cs.map((c, k) => (k ? 'L' : 'M') + n1(lx(c)) + ' ' + n1(by(M.cv_acc[k]))).join(''), { stroke: C.frame, sw: 1.2 });
  M.Cs.forEach((c, k) => {
    const chosen = Math.abs(c - M.C) / M.C < 1e-3;
    out.push({ k: 'cvp' + k, t: 'circle', a: { cx: lx(c), cy: by(M.cv_acc[k]), r: chosen ? 6 : 3.8, fill: chosen ? C.accent : C.surface, stroke: chosen ? C.accent : C.ink, 'stroke-width': 1.4 },
      delay: k * 50, tip: `C = ${c < .01 || c > 1000 ? c.toExponential(2) : +c.toPrecision(3)}<br>CV accuracy <b>${pct(M.cv_acc[k], 2)}</b>${chosen ? '<br>chosen' : ''}` });
  });
  for (const e of [-4, -2, 0, 2, 4]) T(out, 'cv-x' + e, lx(Math.pow(10, e)), bB + 12, e === 0 ? '1' : '10' + (e < 0 ? '⁻' : '') + ['⁰', '¹', '²', '³', '⁴'][Math.abs(e)], { anchor: 'middle', fs, color: C.muted });
  T(out, 'cv-xl', x1 + 20, bB + 12, 'C →', { anchor: 'start', fs, color: C.muted });
  L(out, 'pa-c', lx(M.C), aT - 6, lx(M.C), bB, C.accent, { sw: 1.5, dash: '4 3' });
  T(out, 'pa-ct', lx(M.C) + 5, aT, `C = ${fx(M.C, 4)}`, { fs: fs + .5, color: C.accent, bold: true });
  T(out, 'pa-l', x0 + 4, aT + 6, '← strong penalty', { fs, color: C.muted, mono: false });
  return out;
}

let LF = 0;
function sceneLearn(d) {
  const out = [], fr = LEARN[LF], last = LEARN.length - 1;
  const wide = d.W >= 700;
  const lb = wide ? { x: d.pad + (d.W - 2 * d.pad) * .62, y: d.pad + 34, w: (d.W - 2 * d.pad) * .38, h: Math.min(220, (d.H - 2 * d.pad) * .5) }
    : { x: d.pad + 30, y: d.pad + 30, w: d.W - 2 * d.pad - 34, h: (d.H - 2 * d.pad) * .18 };
  const bars = wide ? { x: d.pad, y: d.pad + 34, w: (d.W - 2 * d.pad) * .58, h: d.H - 2 * d.pad - 84 }
    : { x: d.pad, y: lb.y + lb.h + 34, w: d.W - 2 * d.pad, h: d.H - lb.y - lb.h - 34 - d.pad - 44 };
  T(out, 'le-h', d.pad, d.pad + 8, d.narrow ? 'Re-enactment (proximal gradient descent)' : 'Re-enactment: proximal gradient descent, same L1 objective', { fs: d.narrow ? 9.5 : 11.5, color: C.muted, mono: false });
  coefBars(out, d, bars, fr.w, fr.b);
  // loss curve on a log iteration axis
  const nums = LEARN.slice(0, last), maxIt = nums[nums.length - 1].it;
  const lxp = it => lb.x + Math.log10(1 + it) / Math.log10(1 + maxIt) * lb.w;
  const lo = .355, hi = .7, ly = v => lb.y + (hi - v) / (hi - lo) * lb.h;
  const k = Math.min(LF, last - 1);
  const pts = nums.map((f, i) => { const g = nums[Math.min(i, k)]; return [lxp(g.it), ly(g.loss)]; });
  L(out, 'lo-ax', lb.x, lb.y + lb.h, lb.x + lb.w, lb.y + lb.h, C.frame);
  L(out, 'lo-ay', lb.x, lb.y, lb.x, lb.y + lb.h, C.frame);
  T(out, 'lo-t1', lb.x - 4, ly(.7), '0.70', { anchor: 'end', fs: 9.5, color: C.muted });
  T(out, 'lo-t2', lb.x - 4, ly(.36), '0.36', { anchor: 'end', fs: 9.5, color: C.muted });
  T(out, 'lo-xl', lb.x + lb.w, lb.y + lb.h + 11, `iterations (log) → ${int(maxIt)}`, { anchor: 'end', fs: 9.5, color: C.muted });
  P(out, 'lo-line', pts.map((p, i) => (i ? 'L' : 'M') + n1(p[0]) + ' ' + n1(p[1])).join(''), { stroke: C.flow, sw: 2 });
  const cp = LF === last ? [lxp(maxIt), ly(fr.loss)] : pts[k];
  out.push({ k: 'lo-pt', t: 'circle', a: { cx: cp[0], cy: cp[1], r: 5, fill: LF === last ? C.accent : C.flow, stroke: C.surface, 'stroke-width': 2 } });
  const info = LF === last ? 'fitted model (scikit-learn)' : `iteration ${int(fr.it)}`;
  const tx = wide ? lb.x : d.pad + (d.W - 2 * d.pad) * .45, ty = wide ? lb.y + lb.h + 34 : d.pad + 26;
  T(out, 'lo-it', wide ? tx : d.W - d.pad, ty, info, { fs: d.narrow ? 10.5 : 12.5, bold: true, color: LF === last ? C.accent : C.ink, anchor: wide ? 'start' : 'end' });
  T(out, 'lo-v', wide ? tx : d.W - d.pad, ty + 18, `training log-loss ${fr.loss.toFixed(4)} · accuracy ${pct(fr.acc)}`, { fs: d.narrow ? 9.5 : 11.5, color: C.muted, anchor: wide ? 'start' : 'end' });
  return out;
}
function playLearn(ctx) {
  clearRun(); LF = 0; S.render(sceneLearn(dims()), RM ? 0 : 300);
  if (RM) { LF = LEARN.length - 1; S.render(sceneLearn(dims()), 0); return; }
  const id = setInterval(() => {
    if (LF >= LEARN.length - 1) { clearInterval(id); return; }
    LF++; S.render(sceneLearn(dims()), 170);
  }, 190);
  TIMERS.push(id);
}

// ================================================================ chapter 5: results
function testGeom(d) {
  const x0 = d.pad + 8, x1 = d.W - d.pad - 8, t0 = d.pad + 40, bot = d.H - d.pad - 30, mid = (t0 + bot) / 2;
  return { x0, x1, won: [t0, mid - 10], lost: [mid + 10, bot], mid, bot };
}
const TEST_P = TE.logit.map(sigmoid);
function sceneTest(d) {
  const out = [], g = testGeom(d), X = p => g.x0 + p * (g.x1 - g.x0), fs = d.narrow ? 10 : 12;
  let tp = 0, fn = 0, fp = 0, tn = 0;
  TE.logit.forEach((z, i) => { const w = z > 0, y = TE.y[i] === 1; if (y && w) tp++; else if (y) fn++; else if (w) fp++; else tn++; });
  R(out, 'tb-w', g.x0, g.won[0], g.x1 - g.x0, g.won[1] - g.won[0], mix(C.zero, C.good, .08), { stroke: C.line, rx: 6 });
  R(out, 'tb-l', g.x0, g.lost[0], g.x1 - g.x0, g.lost[1] - g.lost[0], mix(C.zero, C.bad, .08), { stroke: C.line, rx: 6 });
  T(out, 'tb-wl', g.x0 + 2, g.won[0] - 10, `actually won (${int(tp + fn)})`, { fs, color: C.good, bold: true, mono: false });
  T(out, 'tb-ll', g.x0 + 2, g.lost[0] - 10 + 0, `actually lost (${int(fp + tn)})`, { fs, color: C.bad, bold: true, mono: false, a: { y: g.lost[1] + 12 } });
  L(out, 'tb-th', X(.5), g.won[0] - 6, X(.5), g.lost[1] + 4, C.ink, { sw: 1.5, dash: '5 4' });
  for (const p of [0, .25, .5, .75, 1]) T(out, 'tb-x' + p, X(p), g.bot + 16, pct(p, 0), { anchor: 'middle', fs: 10, color: C.muted });
  T(out, 'tb-xl', X(.5), d.pad + 10, d.narrow ? '← loss · predicted win chance · win →' : '← predicted loss · predicted win chance · predicted win →', { anchor: 'middle', fs: d.narrow ? 9.5 : 11, color: C.muted, mono: false });
  const q = (k, x, y, txt, anchor, color, tip) => T(out, k, x, y, txt, { fs: d.narrow ? 10 : 12, bold: true, anchor, color, tip, delay: 900 });
  const ys = (b, f) => b[0] + (b[1] - b[0]) * f;
  q('tq-tp', g.x1 - 6, ys(g.won, .18), `${int(tp)} ✓`, 'end', C.good, `${int(tp)} wins predicted as wins`);
  q('tq-fn', g.x0 + 6, ys(g.won, .18), `${int(fn)} missed`, 'start', C.bad, `${int(fn)} wins the model called losses`);
  q('tq-tn', g.x0 + 6, ys(g.lost, .82), `${int(tn)} ✓`, 'start', C.good, `${int(tn)} losses predicted as losses`);
  q('tq-fp', g.x1 - 6, ys(g.lost, .82), `${int(fp)} false alarm`, 'end', C.bad, `${int(fp)} losses the model called wins`);
  return out;
}
function drawDots(d) {
  const cv = S.canvas, dpr = Math.min(2, window.devicePixelRatio || 1), g = testGeom(d);
  cv.width = Math.round(d.W * dpr); cv.height = Math.round(d.H * dpr);
  const c2 = cv.getContext('2d');
  let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const pts = TE.y.map((y, i) => {
    const b = y ? g.won : g.lost, p = TEST_P[i];
    return [g.x0 + 3 + p * (g.x1 - g.x0 - 6), b[0] + 5 + rnd() * (b[1] - b[0] - 10), y];
  });
  const good = C.good, bad = C.bad, r = d.narrow ? 1.25 : 1.6;
  const paint = t => {
    c2.setTransform(dpr, 0, 0, dpr, 0, 0); c2.clearRect(0, 0, d.W, d.H); c2.globalAlpha = .38;
    for (const [x, y, lab] of pts) {
      const yy = g.mid + (y - g.mid) * t;
      c2.fillStyle = lab ? good : bad; c2.beginPath(); c2.arc(x, yy, r, 0, 6.2832); c2.fill();
    }
  };
  cv.classList.add('on');
  if (RM) { paint(1); return; }
  const t0 = performance.now();
  const loop = now => { const e = clamp((now - t0) / 900, 0, 1); paint(EASE(e)); if (e < 1) ANIM = requestAnimationFrame(loop); else ANIM = 0; };
  ANIM = requestAnimationFrame(loop);
}
function canvasOff() { S.canvas.classList.remove('on'); }

function sceneScore(d) {
  const out = [], wide = d.W >= 640, fs = d.narrow ? 10.5 : 12;
  const box = wide ? { x: d.pad + 70, y: d.pad + 46, s: Math.min((d.W - 2 * d.pad) * .42 - 70, d.H - 2 * d.pad - 120) / 2 }
    : { x: d.pad + 52, y: d.pad + 34, s: Math.min((d.W - 2 * d.pad - 52) / 2, (d.H - 2 * d.pad) * .48 / 2 - 14) };
  const max = Math.max(TP, FN, FP, TN);
  const cells = [[TP, 0, 0, true, 'won, predicted win'], [FN, 0, 1, false, 'won, predicted loss'], [FP, 1, 0, false, 'lost, predicted win'], [TN, 1, 1, true, 'lost, predicted loss']];
  T(out, 'cm-h0', box.x + box.s / 2, box.y - 10, 'pred. win', { anchor: 'middle', fs: fs - .5, color: C.muted, mono: false });
  T(out, 'cm-h1', box.x + box.s * 1.5, box.y - 10, 'pred. loss', { anchor: 'middle', fs: fs - .5, color: C.muted, mono: false });
  T(out, 'cm-act', box.x - 8, box.y - 10, 'actual', { anchor: 'end', fs: fs - .5, color: C.muted, mono: false });
  T(out, 'cm-r0', box.x - 8, box.y + box.s / 2, 'won', { anchor: 'end', fs, color: C.good, bold: true, mono: false });
  T(out, 'cm-r1', box.x - 8, box.y + box.s * 1.5, 'lost', { anchor: 'end', fs, color: C.bad, bold: true, mono: false });
  cells.forEach(([n, r, c, ok, lab], i) => {
    const fill = mix(C.zero, ok ? C.good : C.bad, .2 + .6 * n / max);
    cell(out, 'cm' + i, box.x + c * box.s + 2, box.y + r * box.s + 2, box.s - 4, box.s - 4, int(n), { fill, bold: true, rx: 6,
      fs: clamp(box.s * .2, 12, 26), delay: i * 90, tip: `<b>${int(n)}</b> test rows: ${lab}` });
  });
  const acc = (TP + TN) / (TP + FN + FP + TN);
  const my = box.y + 2 * box.s + 22;
  T(out, 'sc-acc', box.x - (wide ? 70 : 52) + 2, my, `accuracy ${pct(acc, 2)}`, { fs: d.narrow ? 13 : 15, bold: true });
  T(out, 'sc-sub', box.x - (wide ? 70 : 52) + 2, my + 20, `train ${pct(met.train_acc, 2)} · ROC AUC ${met.auc.toFixed(4)}`, { fs: fs - 1, color: C.muted });
  // the notebook's model comparison (its saved table)
  const cx = wide ? d.pad + (d.W - 2 * d.pad) * .48 : d.pad, cw = wide ? (d.W - 2 * d.pad) * .52 : d.W - 2 * d.pad;
  const cy = wide ? d.pad + 46 : my + 62, rows = M.notebook_table, rh = Math.min(42, ((wide ? d.H - d.pad - 30 : d.H - d.pad - 16) - cy) / rows.length);
  T(out, 'mc-h', cx, cy - 18, 'test accuracy of the notebook’s 4 models', { fs: fs - .5, color: C.muted, mono: false });
  const a0 = .5, a1 = .9, bx0 = cx + 2, bx1 = cx + cw - 2, AX = a => bx0 + (a - a0) / (a1 - a0) * (bx1 - bx0);
  rows.forEach((m, j) => {
    const y = cy + j * rh, fin = !!m.final;
    const nm = d.narrow ? ['Baseline', 'LASSO L1 (final)', 'Ridge L2', 'RFECV'][j] || m.name : m.name;
    T(out, 'mc-n' + j, bx0, y + rh * .22, `${nm} · ${m.nonzero} weights`, { fs: Math.min(fs - 1, fsz(rh * .45, cw, m.name + ' · 57 weights')), color: fin ? C.accent : C.ink, bold: fin, mono: false });
    R(out, 'mc-b' + j, bx0, y + rh * .42, AX(m.test) - bx0, rh * .4, fin ? C.accent : C.frame, { rx: 2, from: { width: 0 }, delay: 300 + j * 100,
      tip: `${m.name}<br>test ${pct(m.test, 2)} · train ${pct(m.train, 2)} · AUC ${m.auc.toFixed(4)}<br>${m.nonzero} non-zero weights` });
    T(out, 'mc-v' + j, AX(m.test) + 5, y + rh * .62, pct(m.test, 2), { fs: fs - 1, color: fin ? C.accent : C.muted, bold: fin });
  });
  const ay = cy + rows.length * rh + 2;
  L(out, 'mc-ax', bx0, ay, bx1, ay, C.frame);
  T(out, 'mc-a0', bx0, ay + 10, '50% (coin flip)', { fs: 9.5, color: C.muted, mono: false });
  T(out, 'mc-a1', bx1, ay + 10, '90%', { anchor: 'end', fs: 9.5, color: C.muted });
  return out;
}

// ================================================================ chapter 6: try it
const TRYBASE = { gold: (T1.at25.goldat25 + T1.at25.opp_goldat25) / 2, xp: (T1.at25.xpat25 + T1.at25.opp_xpat25) / 2, cs: (T1.at25.csat25 + T1.at25.opp_csat25) / 2 };
const TRY0 = { gl: T1.at25.golddiffat25, xl: T1.at25.xpdiffat25, cl: T1.at25.csdiffat25, k: T1.at25.killsat25, dt: T1.at25.deathsat25, side: 1 };
let TRY = Object.assign({}, TRY0);
function tryEval(s) {
  const x = { goldat25: TRYBASE.gold + s.gl / 2, opp_goldat25: TRYBASE.gold - s.gl / 2, xpat25: TRYBASE.xp + s.xl / 2, opp_xpat25: TRYBASE.xp - s.xl / 2,
    csat25: TRYBASE.cs + s.cl / 2, opp_csat25: TRYBASE.cs - s.cl / 2, killsat25: s.k, deathsat25: s.dt, side_binary: s.side, gamelength: T1.gamelength };
  const prods = NUM.map((c, i) => M.coef[i] * (x[c] - M.mean[i]) / M.scale[i]);
  const logit = prods.reduce((a, b) => a + b, M.intercept);
  return { x, prods, logit, p: sigmoid(logit) };
}
function sceneTry(d) {
  const out = [], r = tryEval(TRY), wide = d.W >= 640;
  const cb = wide ? { x: d.pad, y: d.pad + 18, w: (d.W - 2 * d.pad) * .5, h: d.H - 2 * d.pad - 40 }
    : { x: d.pad, y: d.pad + 10, w: d.W - 2 * d.pad, h: (d.H - 2 * d.pad) * .5 };
  contrib(out, d, cb, [{ key: 'cb1', prods: r.prods, color: C.accent, sign: true, label: 'your team' }], { max: 1.6 });
  const g = wide ? { x0: d.pad + cb.w + 52, x1: d.W - d.pad - 6, y1: d.pad + 64, y0: d.H - d.pad - 24 }
    : { x0: d.pad + 24, x1: d.W - d.pad - 4, y1: cb.y + cb.h + 20, y0: d.H - d.pad - 18 };
  sigPlot(out, g, d, { noFormula: true });
  sigPoint(out, 'pt1', g, r.logit, r.p >= .5 ? C.good : C.bad, '');
  const big = wide ? 30 : 18, px = wide ? g.x0 + 8 : g.x0 + 8, py = wide ? d.pad + 22 : g.y1 + 26;
  T(out, 'tr-p', px, py, `${pct(r.p)} win`, { fs: big, bold: true, color: r.p >= .5 ? C.good : C.bad });
  T(out, 'tr-z', px, py + big * .5 + 10, `logit ${fx(r.logit, 3)}`, { fs: wide ? 12 : 10, color: C.muted,
    tip: `your gold ${int(r.x.goldat25)} vs ${int(r.x.opp_goldat25)}<br>XP ${int(r.x.xpat25)} vs ${int(r.x.opp_xpat25)}<br>CS ${fx(r.x.csat25, 1)} vs ${fx(r.x.opp_csat25, 1)}` });
  return out;
}
const SLIDERS = [['gl', 'Gold lead', -15000, 15000, 1], ['xl', 'XP lead', -12000, 12000, 1], ['cl', 'CS lead', -150, 150, 1], ['k', 'Kills', 0, 30, 1], ['dt', 'Deaths', 0, 30, 1]];
const TRY_BODY = `<p>Drag the sliders. This runs the real fitted model live: scale each input, multiply by its weight, add, squash. It starts at T1’s real row, so it reads ${pct(T1.p)}.</p>
<div class="lol-try">${SLIDERS.map(([k, lab, a, b, st]) => `<label><span>${lab}</span><output id="o-${k}"></output><input type="range" id="s-${k}" min="${a}" max="${b}" step="${st}" aria-label="${lab}"></label>`).join('')}
<div class="seg"><span>Side</span><button type="button" data-side="0">Blue</button><button type="button" data-side="1" class="red">Red</button></div>
<div class="row"><button type="button" class="btn" id="try-reset">Reset to T1’s row</button></div></div>
<p class="note">Totals stay at this game’s level (gold ${int(TRYBASE.gold)} each, XP ${int(TRYBASE.xp)}, CS ${fx(TRYBASE.cs, 1)}); a lead adds half to you and takes half from the enemy.</p>`;
function wireTry() {
  const $ = id => document.getElementById(id);
  const sync = () => {
    for (const [k] of SLIDERS) { const el = $('s-' + k); if (!el) return; el.value = TRY[k]; $('o-' + k).textContent = /l$/.test(k) ? (TRY[k] > 0 ? '+' : '') + int(TRY[k]) : String(TRY[k]); }
    document.querySelectorAll('.lol-try [data-side]').forEach(b => b.classList.toggle('on', +b.dataset.side === TRY.side));
  };
  const upd = () => { sync(); S.render(sceneTry(dims()), RM ? 0 : 160); };
  for (const [k] of SLIDERS) { const el = $('s-' + k); if (el) el.addEventListener('input', () => { TRY[k] = +el.value; upd(); }); }
  document.querySelectorAll('.lol-try [data-side]').forEach(b => b.addEventListener('click', () => { TRY.side = +b.dataset.side; upd(); }));
  const rs = $('try-reset'); if (rs) rs.addEventListener('click', () => { TRY = Object.assign({}, TRY0); upd(); });
  sync();
}

// ================================================================ steps
let CUR = null, TIMERS = [], ANIM = 0;
function clearRun() {
  TIMERS.forEach(t => { clearTimeout(t); clearInterval(t); }); TIMERS = [];
  if (ANIM) cancelAnimationFrame(ANIM); ANIM = 0;
}
function mk(title, body, build, x = {}) {
  return {
    title, body,
    enter(ctx) {
      clearRun(); CUR = { build, x };
      if (!x.canvas) canvasOff();
      S.render(build(dims()), RM ? 0 : ctx.from < 0 ? 600 : (x.dur ?? 750));
      if (x.enter) x.enter(ctx);
    },
    leave() { clearRun(); },
  };
}
function redraw(dur) {
  if (!CUR) return;
  S.render(CUR.build(dims()), RM ? 0 : dur);
  if (CUR.x.canvas) { if (ANIM) cancelAnimationFrame(ANIM); ANIM = 0; const keep = RM; RM = true; drawDots(dims()); RM = keep; }
}

const T0 = TEAMS[0];
const w3 = c => fx(W_OF(c), 3);
const chapters = [
  { title: 'The data', steps: [
    mk('Twelve rows per game', `<p>The project starts from Oracle’s Elixir’s file of 2024 pro League of Legends games: <b>${int(cnt.rows)}</b> rows and ${cnt.cols} columns covering ${int(cnt.games)} games.</p>
<p>Each game takes 12 rows: one per player (10) and one per team (2). We’ll follow one real game all the way: <b>game ${GM.game} of the 2024 Worlds final</b>, ${T0.team} (blue side) vs ${T1.team} (red side).</p>
<p class="note">Hover or tap any cell to see its exact value and where it came from.</p>`, centered(sceneRaw(0), false)),
    mk('Keep the two team rows', `<p>The question is about teams, so the notebook keeps only rows where <code>position == 'team'</code>. That leaves <b>${int(cnt.team_rows)}</b> rows, two per game.</p>
<p>A team row’s numbers are the sums of its five players. Hover T1’s gold to check.</p>
<pre>data = data[data['position'] == 'team']</pre>`, centered(sceneRaw(1), false)),
    mk('Drop games with no minute-25 snapshot', `<p>Not every game recorded stats at minute 25. In ${pct(cnt.missing_frac)} of team rows, all 15 “at 25” columns are empty together, so the notebook drops those rows instead of guessing.</p>
<p>Most of the ${int(cnt.missing_rows)} dropped rows are LPL and LDL games. <b>${int(cnt.clean_rows)}</b> rows remain. Our game has full data, so it stays.</p>
<pre>data = data.dropna(subset=columns_to_check)</pre>`, centered(sceneRaw(2), false)),
    mk('A snapshot at minute 25', `<p>Here are the 15 minute-25 columns for our game, one column per team. <span style="color:var(--pos)">Orange</span> means more than the other team, <span style="color:var(--neg)">blue</span> means less.</p>
<p>It was close: at 25:00 T1 led by only ${int(T1.at25.golddiffat25)} gold. And the arrows show the mirror: ${T0.team}’s <code>goldat25</code> is T1’s <code>opp_goldat25</code>.</p>`, centered(sceneSnap)),
    mk('Does a gold lead win games?', `<p>All ${int(cnt.clean_rows)} clean rows, by gold lead at minute 25: <span style="color:var(--good)">wins</span> above the line, <span style="color:var(--bad)">losses</span> below.</p>
<p>Teams ahead in gold won <b>${pct(cnt.ahead_win_rate)}</b> of the time. Our game sits right in the middle at ±${int(T1.at25.golddiffat25)}: on gold alone, nearly a coin flip.</p>`, sceneHist),
  ] },
  { title: 'Features', steps: [
    mk('Pick the model’s inputs', `<p>The final model gets 11 columns: own gold, XP, CS, kills and deaths; the opponent’s gold, XP and CS; side; game length; and league.</p>
<p>Leads and the opponent’s kills or deaths are left out: they repeat information already there (gold lead = gold − opp gold). Side becomes a number: Blue = 0, Red = 1.</p>
<p class="note"><code>gamelength</code> is the final length of the game, which nobody knows at minute 25. Luckily the model ends up giving it a weight of exactly 0.</p>`, centered(sceneInputs)),
    mk('Split: learn on 70%, test on 30%', `<p>Rows are dealt at random: <b>${int(met.n_train)}</b> for training and <b>${int(met.n_test)}</b> held back for the final exam. Both rows of our game landed in training.</p>
<p>The split is by row, not by game, so in ${int(cnt.games_split)} of ${int(cnt.games_clean)} games one team is in training and the other team’s mirror row is in test.</p>
<p class="note">The README says the split was stratified; the notebook’s call has no <code>stratify</code> option, and this page follows the notebook.</p>`, sceneSplit),
    mk('Put every input on the same scale', `<p>Gold is in the tens of thousands, kills are single digits. <code>StandardScaler</code> rewrites each input as z = (x − mean) / std, using the training rows.</p>
<p>Now 0 means “average” and +1 means “one typical spread above average”. T1’s ${int(T1.x[IDX.csat25])} CS is ${fx(T1.z[IDX.csat25], 2)} spreads above average.</p>`, centered(sceneScale)),
    mk('League becomes 47 switches', `<p>League is a name, not a number, so <code>OneHotEncoder</code> turns it into 47 on/off columns, one per league in the training data. For our game only <code>league_${T1.league}</code> is on.</p>
<p>That makes <b>10 + 47 = 57</b> numbers per row: the model’s whole view of the game.</p>`, centered(sceneOneHot)),
  ] },
  { title: 'Logistic regression', steps: [
    mk('Multiply by the weights', `<p>The trained model has one weight per input. Multiply each scaled input by its weight. Positive products push toward “win”, negative ones toward “loss”.</p>
<p>Biggest push for T1: enemy gold below average (z = ${fx(T1.z[IDX.opp_goldat25], 2)}) times a negative weight (${w3('opp_goldat25')}) gives <b>${sg(T1.prod[IDX.opp_goldat25], 3)}</b>.</p>`, centered(sceneFwd(false))),
    mk('Add them up: the logit', `<p>Add all 57 products, then the intercept. The total is the <b>logit</b>, the model’s raw score: above 0 leans win, below 0 leans loss.</p>
<p>The L1 penalty pushed the intercept and all 47 league weights to exactly 0, so only 9 products count. T1’s logit is <b>${fx(T1.logit, 3)}</b>.</p>`, centered(sceneFwd(true))),
    mk('Squash it into a probability', `<p>The <b>sigmoid</b> bends any score into a number between 0 and 1: σ(z) = 1 / (1 + e<sup>−z</sup>).</p>
<p>A logit of ${fx(T1.logit, 3)} becomes <b>${T1.p.toFixed(3)}</b>: a ${pct(T1.p)} chance that T1 wins. That is above 0.5, so the model predicts a win, and T1 did win the game and the title.</p>`, sceneSig, { dur: 1200 }),
    mk('Same math for the other row', `<p>${T0.team}’s row goes through the same steps: logit ${fx(T0.logit, 3)}, so <b>${pct(T0.p)}</b>.</p>
<p>The model scores each row on its own, so the two chances add up to ${pct(T0.p + T1.p)}, not exactly 100%. The bars show what pushed each team’s score up or down.</p>`, sceneBoth),
  ] },
  { title: 'Training', steps: [
    mk('What the weights say', `<p>These are the fitted weights (on scaled inputs). Own gold (${sg(W_OF('goldat25'), 2)}) and enemy gold (${fx(W_OF('opp_goldat25'), 2)}) matter most, then enemy XP and own XP. Kills help, deaths hurt, side barely matters.</p>
<p>Game length and all 47 leagues got exactly 0. LASSO switched them off, leaving <b>${NONZERO}</b> inputs.</p>`, centered(sceneCoef)),
    mk('How strong a penalty?', `<p>LASSO adds a penalty on the size of the weights. <code>LogisticRegressionCV(cv=5, penalty='l1')</code> tried 10 strengths C and scored each with 5-fold cross-validation on the training rows.</p>
<p>Small C means a strong penalty: every weight is 0 (${pct(M.cv_acc[0])} accuracy). It picked <b>C = ${fx(M.C, 4)}</b> (the README’s λ = 1/C = ${M.lambda_readme}).</p>`, scenePath, { dur: 1000 }),
    mk('Watch it learn (a re-enactment)', `<p>Training looks for weights that make the log-loss small without letting weights grow for nothing. Starting from all zeros, gold and enemy gold grow first.</p>
<p class="note">Re-enactment: scikit-learn used liblinear’s solver. Here the same objective is replayed with simple proximal gradient descent; it lands on the fitted weights (largest gap ${M.ista_gap.toFixed(3)}), shown in the last frame.</p>`, sceneLearn,
      { enter(ctx) {
        playLearn(ctx);
        ctx.controls.innerHTML = '<button type="button" class="btn">Replay</button>';
        ctx.controls.firstChild.onclick = () => playLearn(ctx);
      } }),
  ] },
  { title: 'How good is it', steps: [
    mk('Every test row', `<p>Now the ${int(met.n_test)} test rows the model never saw. Each dot is one team row, placed by its predicted win chance.</p>
<p>Top band: teams that won. Bottom band: teams that lost. Right of the dashed line the model says “win”. Most dots end up on the correct side.</p>`, sceneTest,
      { canvas: true, enter() { drawDots(dims()); } }),
    mk('The scorecard', `<p>${int(TP + TN)} of ${int(met.n_test)} test rows correct: <b>${pct(met.test_acc, 2)}</b> accuracy (training: ${pct(met.train_acc, 2)}, ROC AUC ${met.auc.toFixed(4)}).</p>
<p>The 5-input baseline scored ${pct(M.notebook_table[0].test, 2)}. Ridge is a hair higher but keeps all 57 weights; LASSO keeps ${NONZERO}.</p>
<p class="note">${met.matches_notebook ? 'Re-running the notebook’s code and split reproduces its saved numbers exactly.' : 'Shown: the notebook’s saved numbers.'}</p>`, sceneScore),
  ] },
  { title: 'Try it', steps: [
    mk('Try it yourself', TRY_BODY, sceneTry, { enter() { wireTry(); } }),
  ] },
];

await createStepper({
  slug: 'lol-predictor',
  title: 'Checkpoint Champion',
  credit: 'Team project by Xingzhi Cui and Yun Jong (Kevin) Na. Data: Oracle’s Elixir 2024 pro matches',
  repo: REPO,
  chapters,
  init(ctx) {
    readColors();
    RM = !!ctx.reducedMotion;
    S = new Scene(ctx.stage, ctx.tip);
    let pend = 0, lastW = ctx.stage.clientWidth, lastH = ctx.stage.clientHeight;
    new ResizeObserver(() => {
      if (pend) return;
      pend = requestAnimationFrame(() => {
        pend = 0;
        const w = ctx.stage.clientWidth, h = ctx.stage.clientHeight;
        if (Math.abs(w - lastW) < 1 && Math.abs(h - lastH) < 1) return;
        lastW = w; lastH = h; redraw(0);
      });
    }).observe(ctx.stage);
    addEventListener('themechange', () => { readColors(); redraw(300); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); redraw(300); });
  },
});
