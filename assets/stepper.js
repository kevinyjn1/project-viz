// The step-by-step walkthrough shell shared by every project page.
//
// A project page calls:
//   import { createStepper } from '../assets/stepper.js';
//   createStepper({ slug, title, credit, repo, chapters, init });
//
//   chapters: [{ title: 'Audio', steps: [ step, step, ... ] }, ...]
//   step: { title: 'One tap', body: '<p>HTML text for the panel</p>',
//           enter(ctx) {...},   // draw / animate this step on the stage
//           leave(ctx) {...} }  // optional clean-up before the next step
//   init(ctx): optional, runs once before the first step (load data, build the scene).
//
//   ctx = { stage: <div>, tip: { show(x, y, html), hide() }, controls: <div>,
//           index, step, from (index of the previous step, -1 on first load),
//           data: {} (free space shared between steps), reducedMotion }
//   Steps should animate from the current scene to the new one (not redraw from blank),
//   so moving Next and Back feels smooth. The URL hash (#3) remembers the step.
import { PROJECTS } from './projects.js';

// Apply the saved light/dark choice as early as possible (before the page draws).
try { const t = localStorage.getItem('theme'); if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; } catch (e) {}

const GH_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>';

export function topbar(currentSlug, root = '../') {
  const bar = document.createElement('header');
  bar.id = 'topbar';
  const tabs = PROJECTS.map(p =>
    `<a href="${root}${p.slug}/"${p.slug === currentSlug ? ' class="current" aria-current="page"' : ''}>${p.short}</a>`).join('');
  bar.innerHTML = `<a class="home" href="${root}" aria-label="Under the Hood: all projects"${currentSlug ? '' : ' aria-current="page"'}><span class="long">Under the Hood</span><span class="ico" aria-hidden="true">⌂</span></a>
    <nav id="tabs" aria-label="Projects">${tabs}</nav>
    <button id="theme" type="button" title="Switch light / dark" aria-label="Switch between light and dark colors">◐</button>`;
  bar.querySelector('#theme').onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', document.documentElement.dataset.theme); } catch (e) {}
    window.dispatchEvent(new Event('themechange'));
  };
  return bar;
}

export function githubButton(repo, label = 'View on GitHub') {
  return `<a class="btn" href="${repo}" target="_blank" rel="noopener">${GH_ICON}${label}</a>`;
}

// Read a CSS color token, e.g. css('--accent'). Re-read after a 'themechange' event.
export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export async function createStepper({ slug, title, credit = '', repo, chapters, init }) {
  document.body.classList.add('walk');
  document.body.prepend(topbar(slug));
  const app = document.createElement('div');
  app.id = 'app';
  app.innerHTML = `
    <aside id="panel">
      <header>
        <h2>${title}</h2>
        ${credit ? `<p class="credit">${credit}</p>` : ''}
        <div>${githubButton(repo)}</div>
        <nav id="chapters" aria-label="Chapters"></nav>
      </header>
      <article id="step" aria-live="polite">
        <p class="eyebrow" id="step-chapter"></p>
        <h1 id="step-title"></h1>
        <div id="step-body"></div>
      </article>
      <footer>
        <button id="back" class="btn" type="button">Back</button>
        <span id="count" class="mono"></span>
        <button id="next" class="btn primary" type="button">Next</button>
      </footer>
    </aside>
    <main id="stage"><div id="stage-inner"></div><div id="stage-controls"></div><div class="tip" role="status"></div></main>`;
  document.body.append(app);

  const flat = [];
  chapters.forEach((c, ci) => c.steps.forEach(s => flat.push({ ...s, ci, chapter: c.title })));
  const $ = id => document.getElementById(id);
  const tipEl = app.querySelector('.tip');
  const ctx = {
    stage: $('stage-inner'), controls: $('stage-controls'), data: {}, index: 0, step: null,
    reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
    tip: {
      show(x, y, html) {
        tipEl.innerHTML = html; tipEl.classList.add('show');
        const r = $('stage').getBoundingClientRect();
        const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
        tipEl.style.left = Math.min(Math.max(4, x + 12), r.width - w - 4) + 'px';
        tipEl.style.top = Math.min(Math.max(4, y + 12), r.height - h - 4) + 'px';
      },
      hide() { tipEl.classList.remove('show'); },
    },
  };

  $('chapters').innerHTML = chapters.map((c, i) => `<button type="button" data-c="${i}">${c.title}</button>`).join('');
  $('chapters').onclick = e => {
    const b = e.target.closest('button'); if (!b) return;
    go(flat.findIndex(s => s.ci === +b.dataset.c));
  };

  let cur = -1, ready = false;   // ignore Next/Back/keys until init() has finished loading
  function go(i) {
    if (!ready || !flat.length || !Number.isFinite(i)) return;
    i = Math.max(0, Math.min(flat.length - 1, i));
    if (i === cur) return;
    if (cur >= 0 && flat[cur].leave) {
      try { flat[cur].leave(ctx); } catch (err) { console.error(`Leaving step ${cur + 1} failed:`, err); }
    }
    ctx.tip.hide();
    ctx.controls.innerHTML = '';
    const s = flat[i];
    const from = cur; cur = i;
    ctx.index = i; ctx.step = s; ctx.from = from;
    $('step-chapter').textContent = `${s.ci + 1}. ${s.chapter}`;
    $('step-title').textContent = s.title;
    $('step-body').innerHTML = s.body || '';
    $('step').scrollTop = 0;
    $('count').textContent = `${i + 1} / ${flat.length}`;
    $('back').disabled = i === 0;
    $('next').disabled = i === flat.length - 1;
    [...$('chapters').children].forEach((b, k) => b.classList.toggle('current', k === s.ci));
    history.replaceState(null, '', '#' + (i + 1));
    if (s.enter) {
      try { s.enter(ctx); } catch (err) { console.error(`Step ${i + 1} failed:`, err); }
    }
  }
  $('back').onclick = () => go(cur - 1);
  $('next').onclick = () => go(cur + 1);
  addEventListener('keydown', e => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.defaultPrevented) return;
    const t = e.target;
    if (t && t.closest && (t.closest('input,select,textarea,[contenteditable]'))) return;
    if (e.key === 'ArrowRight') go(cur + 1);
    if (e.key === 'ArrowLeft') go(cur - 1);
  });
  // swipe on the panel text area for phones
  // (only clearly sideways swipes count, so scrolling the text up and down never changes step)
  let sx = null, sy = null;
  $('step').addEventListener('touchstart', e => {
    if (e.touches.length !== 1) { sx = null; return; }
    sx = e.touches[0].clientX; sy = e.touches[0].clientY;
  }, { passive: true });
  $('step').addEventListener('touchend', e => {
    if (sx == null) return;
    const dx = e.changedTouches[0].clientX - sx, dy = e.changedTouches[0].clientY - sy; sx = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) go(cur + (dx < 0 ? 1 : -1));
  }, { passive: true });
  // a link like <a href="#5"> in the text, or editing the address bar, jumps to that step
  addEventListener('hashchange', () => {
    const n = parseInt(location.hash.slice(1), 10);
    if (Number.isFinite(n)) go(n - 1);
  });

  if (init) {
    try { await init(ctx); } catch (err) {
      console.error('Setup failed:', err);
      const msg = document.createElement('p');
      msg.className = 'stage-error';
      msg.textContent = 'Something went wrong while loading this page. Try reloading.';
      $('stage').append(msg);
    }
  }

  ready = true;
  const start = parseInt(location.hash.slice(1), 10);
  go(Number.isFinite(start) ? start - 1 : 0);
  return { go, ctx };
}
