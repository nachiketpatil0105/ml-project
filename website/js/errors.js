'use strict';
document.addEventListener('site-ready', () => {
  const { el, table, fmt } = SITE, r = SITE.data.final, e = SITE.data.errors, c = r.primary_tuned.confusion;
  table(document.getElementById('confusion'), ['', 'Model says keep', 'Model says skip'], [['Track was kept', fmt.int(c[0][0]), fmt.int(c[0][1])], ['Track was skipped', fmt.int(c[1][0]), fmt.int(c[1][1])]], { numeric: [1, 2], caption: 'Every prediction row in the test sessions' });
  const head = ['Group', 'Rows', 'Share skipped', 'Model accuracy', 'Copy-last accuracy'];
  const row = (name, x) => [name, fmt.int(x.n), fmt.pct(x.skip_rate), fmt.pct(x.model), fmt.pct(x.copy_last)];
  table(document.getElementById('by-history'), head, e.by_history.map(x => row(x.bin, x)), { numeric: [1, 2, 3, 4], caption: 'Grouped by how many first-half tracks were skipped' });
  table(document.getElementById('by-context'), head, e.by_context.map(x => row(x.context.replace(/_/g, ' '), x)), { numeric: [1, 2, 3, 4], caption: 'Grouped by where the last history track was played from (at least 200 rows)' });
  table(document.getElementById('by-premium'), head, e.by_premium.map(x => row(x.premium ? 'Premium' : 'Free', x)), { numeric: [1, 2, 3, 4] });
  const pos = e.by_position;
  document.getElementById('position-text').textContent = 'Accuracy is ' + fmt.pct(pos[0].model) + ' for the first position and ' + fmt.pct(pos[pos.length - 1].model) + ' for position ' + pos[pos.length - 1].position + '. The model is ' + (pos.every(p => p.model > p.copy_last) ? 'above' : 'not always above') + ' the copy-last rule at every position shown.';
  const gap = Math.max(...e.calibration.map(x => Math.abs(x.mean_prob - x.observed)));
  document.getElementById('calibration-text').textContent = 'The biggest gap between predicted and real skip rate in any bin is ' + fmt.pct(gap) + '.';
  const host = document.getElementById('examples');
  e.examples.forEach(ex => {
    const box = el('div'); box.append(el('h3', ex.title));
    box.append(el('p', ex.why + ' Share of the first half skipped: ' + fmt.pct(ex.history.reduce((a, b) => a + b, 0) / ex.history.length) + '.'));
    box.append(el('p', 'First half (what the model saw)', 'source-note'));
    const h = el('div', null, 'seq'); ex.history.forEach(v => h.append(el('span', v ? 'skip' : 'kept', v ? 'h-skip' : 'h-keep'))); box.append(h);
    box.append(el('p', 'Second half: what the model said, then what happened (bold border = wrong)', 'source-note'));
    const s = el('div', null, 'seq');
    ex.prob.forEach((p, i) => { const t = el('span', (ex.pred[i] ? 'skip' : 'keep') + ' → ' + (ex.truth[i] ? 'skipped' : 'kept'), ex.pred[i] === ex.truth[i] ? 'ok' : 'bad'); t.title = 'p(skip) = ' + p.toFixed(2); s.append(t); });
    box.append(s); host.append(box);
  });
});
