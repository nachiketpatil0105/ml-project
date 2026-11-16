'use strict';
document.addEventListener('site-ready', () => {
  const { el, table, params, fmt, verdict } = SITE, r = SITE.data.final, d = SITE.data.dev;
  const order = ['global_majority', 'history_majority', 'copy_last', 'popularity_lr', 'history_lr', 'random_forest', 'xgboost', 'catboost', 'lightgbm'];
  const rows = {}; r.rows.forEach(x => { rows[x.id] = x; });
  table(document.getElementById('model-table'), ['Model', 'Test MAA', '95% interval', 'First-prediction accuracy', 'AUC', 'Development CV MAA'],
    order.map(i => [rows[i].name, fmt.f3(rows[i].maa), fmt.f3(rows[i].ci.low) + ' – ' + fmt.f3(rows[i].ci.high), fmt.f3(rows[i].first_accuracy), fmt.f3(rows[i].roc_auc), fmt.f3(rows[i].dev_mean_fold_maa)]),
    { numeric: [1, 3, 4, 5], best: order.indexOf(r.primary), caption: 'Every model was scored once on the ' + fmt.int(r.test_sessions) + ' test sessions. Bold row: the main model.' });
  const names = { copy_last: 'Copy last skip', history_majority: 'First-half skip majority', history_lr: 'Logistic regression', random_forest: 'Random forest', popularity_lr: 'Popularity-only logistic regression' };
  table(document.getElementById('compare-table'), ['LightGBM minus ...', 'Difference in MAA', '95% interval', 'Reading'],
    Object.entries(r.comparisons_vs_primary_tuned).map(([k, v]) => [names[k], fmt.s3(v.difference), fmt.s3(v.low) + ' to ' + fmt.s3(v.high), verdict(v)]), { numeric: [1], caption: 'Paired by session, 2,000 resamples' });
  const lab = { P: 'Popularity only', H: 'History only', PH: 'Popularity + history', PHA: 'Popularity + history + audio', PHS: 'Popularity + history + similarity', PHAS: 'All groups' };
  table(document.getElementById('ablation-table'), ['Features used', 'Number of features', 'Development CV MAA', 'Spread between folds (SD)', 'AUC'],
    d.ablation.map(a => [lab[a.groups], String(a.features), fmt.f4(a.mean_fold_maa), fmt.f4(a.std_fold_maa), fmt.f3(a.roc_auc)]), { numeric: [1, 2, 3, 4], caption: 'Same LightGBM settings every time; only the inputs change' });
  const imp = document.getElementById('importance-list'), names2 = { H: 'History', S: 'Similarity', A: 'Audio and metadata', P: 'Popularity' };
  Object.entries(r.group_importance).sort((a, b) => b[1] - a[1]).forEach(([g, v]) => imp.append(el('li', names2[g] + ': ' + fmt.pct(v))));
  const top = document.getElementById('top-features');
  r.top_features.slice(0, 6).forEach(t => top.append(el('li', t.feature + ' (' + fmt.pct(t.share) + ')')));
});
