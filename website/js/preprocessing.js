'use strict';
document.addEventListener('site-ready', () => {
  const { el, table, fmt } = SITE, p = SITE.data.prep;
  table(document.getElementById('groups-table'), ['Group', 'Name', 'Features', 'What it contains'],
    Object.entries(p.groups).map(([g, v]) => [g, v.name, String(v.count), v.description]), { numeric: [2], caption: 'The four feature groups (' + p.feature_count + ' features in total)' });
  const list = document.getElementById('allowed-columns');
  Object.entries(p.groups).forEach(([g, v]) => {
    const d = el('details'); d.append(el('summary', v.name + ' (' + v.count + ')')); d.append(el('p', v.columns.join(', '), 'source-note')); list.append(d);
  });
  const ex = p.example, box = document.getElementById('example');
  const hist = el('div', null, 'seq');
  ex.history_skip_2.forEach(v => hist.append(el('span', v ? 'skip' : 'kept', v ? 'h-skip' : 'h-keep')));
  box.append(el('p', 'A session with ' + ex.session_length + ' tracks. The first ' + ex.history_skip_2.length + ' tracks are the history (what the listener did):', 'source-note'), hist);
  box.append(el('p', 'We now predict track number ' + ex.row_position + '. This is what one row looks like (a few of the ' + p.feature_count + ' features):', 'source-note'));
  const labels = { L: 'Session length', k: 'Tracks in the history', pos: 'Position in the prediction window', relpos: 'Relative position', h_skip2: 'Share of history tracks skipped', last1: 'Was the last history track skipped? (1 = yes)', last2: 'Second-to-last skipped?', last3: 'Third-to-last skipped?', streak_skip: 'Skips in a row at the end of the history', re_trackdone: 'Share of history tracks that played to the end', t_us_popularity_estimate: 'Popularity of this track', t_energy: 'Energy of this track', t_danceability: 'Danceability of this track', t_valence: 'Valence (mood) of this track', t_tempo: 'Tempo of this track', euc_keep: 'Distance to the kept tracks (audio)', euc_skip: 'Distance to the skipped tracks (audio)' };
  const rows = Object.entries(ex.features).map(([k, v]) => [labels[k] || k, v === null ? 'missing' : (Number.isInteger(v) ? String(v) : v.toFixed(3))]);
  rows.push(['Target: was this track skipped? (skip_2)', ex.target_skip_2 ? 'yes (1)' : 'no (0)']);
  const holder = el('div'); box.append(holder);
  table(holder, ['Feature', 'Value'], rows, { numeric: [1] });
  const m = p.missing_share;
  document.getElementById('missing-text').textContent = fmt.pct(m.euc_keep) + ' of rows have no kept-track profile and ' + fmt.pct(m.euc_skip) + ' have no skipped-track profile.';
});
