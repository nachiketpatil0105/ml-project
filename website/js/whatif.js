'use strict';
// What-if explorer: re-scores held-out second-half tracks with the exported LightGBM trees
// after the visitor edits which first-half tracks were skipped.
(() => {
  const $ = id => document.getElementById(id);
  let model, data, index = {}, session, flags, selectedRow = 0;

  function nanify(a) { return a.map(v => v === null ? NaN : v); }
  function finite(v) { return v.every(Number.isFinite); }
  function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
  function norm(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }
  function dist(a, b) { return Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0)); }
  function centre(vectors) { return vectors[0].map((_, j) => mean(vectors.map(v => v[j]))); }
  function trailing(f, value) { let c = 0; for (let i = f.length - 1; i >= 0 && f[i] === value; i--) c++; return c; }

  function recompute(row, f) {
    const x = nanify(row.features);
    const set = (name, value) => { x[index[name]] = value; };
    const k = f.length;
    set('h_skip2', mean(f));
    set('last3_rate', mean(f.slice(-3)));
    set('last5_rate', mean(f.slice(-5)));
    set('streak_skip', trailing(f, 1));
    set('streak_keep', trailing(f, 0));
    set('pat3', parseInt(f.slice(-3).join(''), 2));
    set('pat4', parseInt(f.slice(-4).join(''), 2));
    for (let i = 1; i <= 5; i++) set('last' + i, k >= i ? f[k - i] : -1);
    const vectors = session.history_vectors.map(nanify);
    const keep = vectors.filter((v, i) => f[i] === 0 && finite(v));
    const skipped = vectors.filter((v, i) => f[i] === 1 && finite(v));
    const cand = nanify(row.candidate);
    const ok = finite(cand);
    const out = {};
    [['keep', keep], ['skip', skipped]].forEach(([name, group]) => {
      if (ok && group.length) {
        const c = centre(group);
        out['euc_' + name] = dist(cand, c);
        out['cos_' + name] = 1 - cand.reduce((s, v, i) => s + v * c[i], 0) / (norm(cand) * norm(c) + 1e-12);
      } else { out['euc_' + name] = NaN; out['cos_' + name] = NaN; }
    });
    Object.entries(out).forEach(([name, value]) => set(name, value));
    set('d_keep_minus_skip', out.euc_keep - out.euc_skip);
    set('euc_lastkept', ok && keep.length ? dist(cand, keep[keep.length - 1]) : NaN);
    return x;
  }

  function walk(node, x) {
    while (node.f !== undefined) {
      const v = x[node.f];
      let left;
      if (Number.isNaN(v)) {
        left = node.m === 'None' ? 0 <= node.t : node.d === 1;
      } else if (node.m === 'Zero' && v === 0) {
        left = node.d === 1;
      } else {
        left = v <= node.t;
      }
      node = left ? node.l : node.r;
    }
    return node.v;
  }
  function score(x) {
    const raw = model.trees.reduce((s, t) => s + walk(t, x), 0);
    return 1 / (1 + Math.exp(-raw));
  }
  window.whatIfScore = score;
  window.whatIfRecompute = (s, r, f) => { session = s; return recompute(r, f); };

  function el(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }

  function drawHistory() {
    const box = $('wi-history');
    box.replaceChildren();
    flags.forEach((v, i) => {
      const b = el('button', v ? 'skip' : 'kept', 'wi-sq ' + (v ? 'is-skip' : 'is-kept'));
      b.type = 'button';
      b.setAttribute('aria-pressed', String(!!v));
      b.setAttribute('aria-label', 'First-half track ' + (i + 1) + ': ' + (v ? 'skipped' : 'kept') + '. Toggle.');
      b.addEventListener('click', () => { flags[i] = 1 - flags[i]; update(); });
      box.append(b);
    });
  }

  function radar(row) {
    const names = data.radar;
    const kept = session.history_radar.filter((r, i) => flags[i] === 0 && r.every(v => v !== null));
    const skipped = session.history_radar.filter((r, i) => flags[i] === 1 && r.every(v => v !== null));
    const series = [['This track', row.radar, '#a3501c', ''], ['Kept first-half mean', kept.length ? centre(kept) : null, '#1f5f8b', ''], ['Skipped first-half mean', skipped.length ? centre(skipped) : null, '#6b7780', '4 3']];
    const size = 260, c = size / 2, r0 = 90, n = names.length;
    const pt = (i, v) => { const a = -Math.PI / 2 + 2 * Math.PI * i / n; return [c + r0 * v * Math.cos(a), c + r0 * v * Math.sin(a)]; };
    let svg = '<svg viewBox="0 0 ' + size + ' ' + size + '" role="img" aria-label="Radar chart of the selected track against the kept and skipped first-half profiles">';
    [0.25, 0.5, 0.75, 1].forEach(t => { svg += '<polygon fill="none" stroke="#d9dfe3" points="' + names.map((_, i) => pt(i, t).join(',')).join(' ') + '"/>'; });
    names.forEach((nm, i) => { const [x, y] = pt(i, 1.18); svg += '<text x="' + x + '" y="' + y + '" font-size="9" text-anchor="middle" dominant-baseline="middle" fill="#58636a">' + nm + '</text>'; });
    series.forEach(([, vals, color, dash]) => { if (vals) svg += '<polygon fill="' + color + '" fill-opacity="0.12" stroke="' + color + '" stroke-width="2" stroke-dasharray="' + dash + '" points="' + vals.map((v, i) => pt(i, Math.max(0, Math.min(1, v))).join(',')).join(' ') + '"/>'; });
    svg += '</svg>';
    $('wi-radar').innerHTML = svg;
    const legend = $('wi-legend');
    legend.replaceChildren();
    series.forEach(([name, vals, color]) => { const s = el('span', name + (vals ? '' : ' (none)')); s.style.borderLeft = '12px solid ' + color; s.style.paddingLeft = '6px'; s.style.marginRight = '14px'; legend.append(s); });
  }

  function update() {
    drawHistory();
    const body = $('wi-rows');
    body.replaceChildren();
    session.rows.forEach((row, j) => {
      const original = row.python_prob;
      const edited = score(recompute(row, flags));
      const tr = el('tr');
      tr.className = j === selectedRow ? 'selected' : '';
      const pick = el('button', 'Track ' + (j + 1), 'link-button');
      pick.type = 'button';
      pick.addEventListener('click', () => { selectedRow = j; update(); });
      const th = el('th'); th.scope = 'row'; th.append(pick);
      tr.append(th);
      [original.toFixed(2), edited.toFixed(2), edited > model.threshold ? 'skip' : 'keep', row.truth ? 'skipped' : 'kept'].forEach((t, i) => { const td = el('td', t); if (i < 2) td.className = 'num'; tr.append(td); });
      const delta = edited - original;
      const dtd = el('td', (delta >= 0 ? '+' : '') + delta.toFixed(2)); dtd.className = 'num'; tr.append(dtd);
      body.append(tr);
    });
    radar(session.rows[selectedRow]);
    const changed = flags.some((v, i) => v !== session.history[i]);
    $('wi-note').textContent = changed ? 'History edited. Probabilities in the "edited" column use the changed first half; every other input stays at its recorded value.' : 'Showing the recorded first half. Click a square to change it.';
  }

  function load() {
    Promise.all([fetch('../../python/06_explorer/results/model.json').then(r => r.json()), fetch('../../python/06_explorer/results/sessions.json').then(r => r.json())]).then(([m, d]) => {
      model = m; data = d;
      m.features.forEach((n, i) => { index[n] = i; });
      const select = $('wi-session');
      d.sessions.forEach((s, i) => { const o = el('option', s.label + ' (' + s.length + ' tracks)'); o.value = i; select.append(o); });
      const choose = () => { session = d.sessions[Number(select.value)]; flags = session.history.slice(); selectedRow = 0; update(); };
      select.addEventListener('change', choose);
      $('wi-reset').addEventListener('click', () => { flags = session.history.slice(); update(); });
      $('wi-status').hidden = true;
      $('wi-app').hidden = false;
      choose();
    }).catch(() => { $('wi-status').textContent = 'The explorer could not load its data files. Check that the site is served over http (for example GitHub Pages).'; });
  }
  if ($('wi-session')) load();
})();
