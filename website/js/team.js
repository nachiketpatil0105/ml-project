'use strict';
document.addEventListener('site-ready', () => {
  const { table, fmt } = SITE, x = SITE.data.extra, base = x.base, sd = base.std_fold_maa;
  const find = id => x.experiments.find(e => e.id === id);
  const reading = e => {
    if (!e.paired) return 'different rows, so not a paired comparison';
    const dir = e.delta_maa > 0 ? 'better' : 'worse', n = e.paired.folds, k = e.delta_maa > 0 ? e.paired.folds_better : n - e.paired.folds_better;
    if (Math.abs(e.delta_maa) >= sd) return 'clearly ' + dir + ': more than one fold SD, ' + dir + ' in ' + k + ' of ' + n + ' folds';
    if (k >= Math.ceil(0.8 * n)) return 'small but consistent: ' + dir + ' in ' + k + ' of ' + n + ' folds';
    return 'no clear effect: well below the fold spread, ' + dir + ' in ' + k + ' of ' + n + ' folds';
  };
  const rows = [['Final model (no change)', fmt.f4(base.mean_fold_maa), '–', '–', 'reference']].concat(
    x.experiments.map(e => [e.change, fmt.f4(e.mean_fold_maa), fmt.s4(e.delta_maa), e.paired ? e.paired.folds_better + ' / ' + e.paired.folds : '–', reading(e)]));
  table(document.getElementById('change-table'), ['Change', 'Development MAA (15 folds)', 'Change in MAA', 'Folds where the change scores higher', 'Reading'], rows,
    { numeric: [1, 2, 3], best: 0, caption: 'One change at a time, same LightGBM settings and same folds as the final model' });
  const t = x.threshold;
  const c = t.curve, near = v => fmt.s4(c[v] - c['0.5']);
  document.getElementById('threshold-line').textContent = 'Threshold: we also changed the decision threshold, without refitting, on the same out-of-fold scores (step 3). ' +
    (t.gain < 1e-9 ? '0.50 is already the best value between 0.40 and 0.60. ' : 'The best value is ' + Number(t.best).toFixed(2) + ', ' + fmt.s4(t.gain) + ' above 0.50. ') +
    'Moving it to 0.48 changes MAA by ' + near('0.48') + ' and to 0.52 by ' + near('0.52') + '. Our rule is to move away from 0.5 only if the gain is at least 0.001, so the threshold used is ' + Number(t.chosen).toFixed(2) + '.';
  const drop = find('drop_reason_shares'), floor = find('rule_floor'), lr = find('model_logistic'), audio = find('drop_A'), rate = find('add_song_skip_rate');
  const paired = x.experiments.filter(e => e.paired), lower = s => s.charAt(0).toLowerCase() + s.slice(1);
  const group = test => paired.filter(test).map(e => lower(e.change)).join('; ');
  const clear = group(e => reading(e).startsWith('clearly')), small = group(e => reading(e).startsWith('small')), none = group(e => reading(e).startsWith('no clear'));
  document.getElementById('change-reading').textContent = 'What this shows. Clear effect: ' + (clear || 'none') + '. Small but consistent effect: ' + (small || 'none') + '. No clear effect: ' + (none || 'none') +
    '. None of the added features improves the model, which is why the final model does not use them.';
  document.getElementById('nk-change').textContent = 'If the history were floor(L/2) songs instead of ceil(L/2), development MAA would be ' + fmt.f4(floor.mean_fold_maa) + ' instead of ' + fmt.f4(base.mean_fold_maa) +
    '. With floor, odd-length sessions lose one history song and gain one song to predict, so the task becomes slightly harder; the rows also differ, so this is not a clean paired comparison. Removing the start/end reason shares changes MAA by ' + fmt.s4(drop.delta_maa) + ' (' + reading(drop) + '). They partly restate the skip history, so the model loses only a little without them.';
  document.getElementById('kp-change').textContent = 'If LightGBM is replaced by logistic regression on popularity and history, development MAA changes by ' + fmt.s4(lr.delta_maa) + ' (' + reading(lr) + '): the gain of the boosted trees comes from combining history features non-linearly. Moving the threshold from 0.50 to 0.52 changes MAA by ' + near('0.52') + ', so 0.5 is kept. Dropping all audio features changes MAA by ' + fmt.s4(audio.delta_maa) + ' (' + reading(audio) + '). Adding a per-song skip rate learned from other sessions changes it by ' + fmt.s4(rate.delta_maa) + ' (' + reading(rate) + ').';
});
