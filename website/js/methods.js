'use strict';
document.addEventListener('site-ready', () => {
  const { el, table, params, fmt } = SITE, d = SITE.data.dev;
  const grid = (id, name) => {
    const g = d.grids[name];
    table(document.getElementById(id), ['Setting', 'CV MAA', 'First accuracy', 'AUC'], g.grid.map(r => [params(r.params), fmt.f3(r.mean_fold_maa), fmt.f3(r.mean_fold_first_accuracy), fmt.f3(r.roc_auc)]), { numeric: [1, 2, 3], best: g.index, caption: 'Each setting was scored with 5-fold cross-validation on the development sessions. The chosen row is in bold.' });
  };
  grid('lr-grid', 'history_lr');
  grid('rf-grid', 'random_forest');
  const names = { lightgbm: 'LightGBM', xgboost: 'XGBoost', catboost: 'CatBoost' };
  table(document.getElementById('space-table'), ['Library', 'What the search could change', 'MAA of the starting setting', 'Best MAA found'],
    Object.keys(names).map(n => [names[n], d.optuna[n].search_space, fmt.f3(d.optuna[n].starting_point_search_maa), fmt.f3(d.optuna[n].best_search_maa)]), { numeric: [2, 3] });
  Object.keys(names).forEach(n => {
    const o = d.optuna[n], best = o.top.findIndex(t => t.trial === o.chosen_trial);
    table(document.getElementById('top-' + n), ['Trial', 'Search MAA', 'Final MAA (15 folds)', 'Settings'], o.top.map(t => [String(t.trial), fmt.f3(t.search_maa), fmt.f3(t.repeated_maa), params(t.params)]), { numeric: [1, 2], best, caption: names[n] + ': the five best trials, scored again on three new fold splits. The chosen trial is in bold.' });
  });
  const order = ['global_majority', 'history_majority', 'copy_last', 'popularity_lr', 'history_lr', 'random_forest', 'lightgbm', 'xgboost', 'catboost'];
  const label = { global_majority: 'Training-label majority', history_majority: 'First-half skip majority', copy_last: 'Copy last skip', popularity_lr: 'Popularity-only logistic regression', history_lr: 'Logistic regression', random_forest: 'Random forest', lightgbm: 'LightGBM', xgboost: 'XGBoost', catboost: 'CatBoost' };
  table(document.getElementById('chosen-table'), ['Model', 'Chosen setting', 'Development CV MAA (15 folds)'], order.map(n => [label[n], params(d.selected[n].params), fmt.f4(d.selected[n].mean_fold_maa)]), { numeric: [2], caption: 'Final choices, all made on development data only' });
});
