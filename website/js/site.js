'use strict';
// Shared script for every page: adds the menu, loads the result files named in <body data-sources>,
// and fills every element that has data-v="path" with the matching number.
(() => {
  const PAGES = [['index.html', 'Home', 'Overview'], ['proposal', 'Project proposal', 'Overview'], ['question', 'Question and problem', 'Problem and data'], ['data', 'Data', 'Problem and data'],
    ['preprocessing', 'Preprocessing', 'Problem and data'], ['methods', 'Baselines and methods', 'Method'], ['evaluation', 'Evaluation', 'Method'], ['results', 'Results and explorer', 'Evidence'],
    ['errors', 'Error analysis', 'Evidence'], ['limitations', 'Limitations', 'Evidence'], ['conclusion', 'Conclusion', 'Evidence'], ['team', 'Team and contributions', 'Submission'], ['reproduction', 'Reproduction, repository and credits', 'Submission']];
  const body = document.body, BASE = body.dataset.base || '', PAGE = body.dataset.page;
  const href = s => s === 'index.html' ? BASE + 'index.html' : (BASE === '' ? 'website/html/' : '') + s + '.html';

  function el(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined && text !== null) n.textContent = text; if (cls) n.className = cls; return n; }

  function buildChrome() {
    const groups = [];
    PAGES.forEach(([slug, label, group]) => { let g = groups.find(x => x.name === group); if (!g) { g = { name: group, items: [] }; groups.push(g); } g.items.push([slug, label]); });
    const header = el('header', null, 'site-header');
    header.innerHTML = '<div class="inner"><a class="brand" href="' + href('index.html') + '">Predicting skips from session behaviour and track features</a>' +
      '<details class="menu"><summary>Menu</summary><nav class="menu-panel" aria-label="Project pages">' +
      groups.map(g => '<div class="menu-group"><p class="menu-title">' + g.name + '</p>' + g.items.map(([s, l]) => '<a href="' + href(s) + '"' + (s === PAGE ? ' aria-current="page"' : '') + '>' + l + '</a>').join('') + '</div>').join('') +
      '</nav></details></div><div class="inner crumb">You are here: ' + (PAGES.find(p => p[0] === PAGE) || ['', ''])[1] + '</div>';
    const skip = el('a', 'Skip to content', 'skip-link'); skip.href = '#main';
    body.prepend(header); body.prepend(skip);
    const main = document.getElementById('main');
    const i = PAGES.findIndex(p => p[0] === PAGE);
    const pager = el('div', null, 'pager');
    pager.innerHTML = (i > 0 ? '<a href="' + href(PAGES[i - 1][0]) + '">&larr; ' + PAGES[i - 1][1] + '</a>' : '<span></span>') + (i >= 0 && i < PAGES.length - 1 ? '<a href="' + href(PAGES[i + 1][0]) + '">' + PAGES[i + 1][1] + ' &rarr;</a>' : '<span></span>');
    main.append(pager);
    const footer = el('footer', 'Skip prediction project · Spotify Music Streaming Sessions dataset (mini partition)');
    body.append(footer);
    const menu = header.querySelector('details.menu');
    document.addEventListener('click', e => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); } });
  }

  const f = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);
  const fmt = {
    f2: v => f(v, 2), f3: v => f(v, 3), f4: v => f(v, 4), pct: v => f(100 * v, 1) + '%', pct0: v => f(100 * v, 0) + '%',
    int: v => Number(v).toLocaleString('en-US'), s3: v => (v >= 0 ? '+' : '') + f(v, 3), s4: v => (v >= 0 ? '+' : '') + f(v, 4), join: v => (v || []).join(', '), raw: v => String(v)
  };

  function lookup(root, path) {
    let cur = root;
    for (const token of path.split('.')) {
      if (cur === undefined || cur === null) return undefined;
      const m = token.match(/^([^\[]+)\[(\w+)=([^\]]+)\]$/);
      if (m) cur = (cur[m[1]] || []).find(x => String(x[m[2]]) === m[3]);
      else cur = cur[token];
    }
    return cur;
  }

  function params(p) {
    if (p === null || p === undefined) return 'none';
    if (typeof p !== 'object') return 'C = ' + p;
    return Object.entries(p).map(([k, v]) => k + ' = ' + (typeof v === 'number' && !Number.isInteger(v) ? Number(v.toPrecision(3)) : v)).join(', ');
  }

  function table(host, headers, rows, opt) {
    opt = opt || {};
    host.replaceChildren();
    const wrap = el('div', null, 'table-wrap'), t = el('table');
    if (opt.caption) t.append(el('caption', opt.caption));
    const thead = el('thead'), hr = el('tr');
    headers.forEach((h, i) => { const th = el('th', h); th.scope = 'col'; if ((opt.numeric || []).includes(i)) th.className = 'num'; hr.append(th); });
    thead.append(hr); t.append(thead);
    const tb = el('tbody');
    rows.forEach((r, ri) => {
      const tr = el('tr'); if (ri === opt.best) tr.className = 'best';
      r.forEach((c, i) => { const td = el('td'); if (c instanceof Node) td.append(c); else td.textContent = c; if ((opt.numeric || []).includes(i)) td.className = 'num'; tr.append(td); });
      tb.append(tr);
    });
    t.append(tb); wrap.append(t); host.append(wrap);
  }

  function verdict(c) {
    if (c.low > 0) return 'clearly better (interval above zero)';
    if (c.high < 0) return 'worse (interval below zero)';
    return 'not clearly different (interval includes zero)';
  }

  function derive(d) {
    const out = {};
    if (d.dev) { const c = d.dev.threshold_curve; out.threshold_gain = Math.max(...Object.values(c)) - c['0.5']; out.best_threshold = Object.keys(c).reduce((a, b) => c[a] >= c[b] ? a : b); out.thr_048 = c['0.48'] - c['0.5']; out.thr_052 = c['0.52'] - c['0.5']; }
    if (d.final) { const t = d.final.primary_tuned; out.ci_half = (t.ci.high - t.ci.low) / 2; out.gain = d.final.comparisons_vs_primary_tuned.copy_last.difference; const p = d.final.test_skip_rate; out.f1_all_skip = 2 * p / (1 + p); }
    if (d.dev && d.final) { out.dev_primary = d.dev.selected[d.final.primary].mean_fold_maa; out.dev_minus_test = out.dev_primary - d.final.primary_tuned.maa; }
    if (d.final) { const m = ['lightgbm', 'xgboost', 'catboost'].map(i => d.final.rows.find(x => x.id === i).maa); out.lib_min = Math.min(...m); out.lib_max = Math.max(...m); out.lib_spread = out.lib_max - out.lib_min; }
    if (d.dev) { const a = {}; d.dev.ablation.forEach(x => { a[x.groups] = x; }); out.ab = a; out.delta_S = a.PHS.mean_fold_maa - a.PH.mean_fold_maa; out.delta_A = a.PHA.mean_fold_maa - a.PH.mean_fold_maa; out.fold_sd = a.PH.std_fold_maa; }
    return out;
  }

  function fill(d) {
    document.querySelectorAll('[data-v]').forEach(n => {
      const v = lookup(d, n.dataset.v);
      if (v === undefined) { n.textContent = '–'; return; }
      n.textContent = (fmt[n.dataset.fmt || 'raw'])(v);
    });
    document.querySelectorAll('[data-href]').forEach(n => { const v = lookup(d, n.dataset.href); if (v) { n.href = v; n.textContent = n.textContent && n.textContent !== '–' ? n.textContent : v; } });
    document.querySelectorAll('[data-if]').forEach(n => { n.hidden = !lookup(d, n.dataset.if) || (Array.isArray(lookup(d, n.dataset.if)) && !lookup(d, n.dataset.if).length); });
    document.querySelectorAll('[data-ifnot]').forEach(n => { const v = lookup(d, n.dataset.ifnot); n.hidden = !!v && !(Array.isArray(v) && !v.length); });
  }

  window.SITE = { el, fmt, table, params, verdict, lookup, data: {}, BASE };
  buildChrome();
  const sources = Object.assign({ config: 'site_config.json' }, JSON.parse(body.dataset.sources || '{}'));
  Promise.all(Object.entries(sources).map(([k, p]) => fetch(BASE + p).then(r => { if (!r.ok) throw new Error(p); return r.json(); }).then(d => { window.SITE.data[k] = d; })))
    .then(() => { window.SITE.data.derived = derive(window.SITE.data); fill(window.SITE.data); document.dispatchEvent(new Event('site-ready')); })
    .catch(err => {
      const box = el('div', null, 'callout');
      box.innerHTML = '<strong>The result files could not be loaded.</strong><p>This site reads its numbers from the <code>python/*/results</code> folders. Open it through a web server (for example run <code>python3 -m http.server</code> in the project folder) or on GitHub Pages. Opening the file directly from disk does not work. Missing file: ' + String(err.message) + '</p>';
      document.getElementById('main').prepend(box);
    });
})();
