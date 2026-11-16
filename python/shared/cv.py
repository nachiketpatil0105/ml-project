"""Grouped cross-validation helpers. Folds are built on development sessions only."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from metrics import evaluate

SEARCH_SEED = 42
REPEAT_SEEDS = [101, 102, 103]


def fold_assignment(keys, y, seed, n_splits=5):
    """Stratified, session-grouped K-fold: all tracks of a session share a fold and the skip rate is balanced across folds."""
    fold = np.full(len(keys), -1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, valid) in enumerate(splitter.split(keys, y, groups=keys.session_id.to_numpy())):
        fold[valid] = f
    return fold


def oof_scores(make, columns, X, y, fold):
    oof = np.zeros(len(X))
    for f in sorted(set(fold)):
        train, valid = fold != f, fold == f
        model = make()
        model.fit(X.loc[train, columns], y[train])
        oof[valid] = model.predict_proba(X.loc[valid, columns])[:, 1]
    return oof


def summarize(keys, y, oof, fold):
    per_fold = []
    for f in sorted(set(fold)):
        m = fold == f
        per_fold.append(evaluate(keys[m].reset_index(drop=True), y[m].reset_index(drop=True), oof[m]))
    return {"mean_maa": float(np.mean([r["maa"] for r in per_fold])), "mean_first_accuracy": float(np.mean([r["first_accuracy"] for r in per_fold])),
            "fold_maa": [r["maa"] for r in per_fold], "roc_auc": float(np.mean([r["roc_auc"] for r in per_fold]))}


def repeated_cv(make, columns, X, y, keys, folds_by_seed):
    """Run K-fold once per seed. Returns the mean over all folds of all repeats plus the repeat-averaged out-of-fold scores."""
    results, oofs = [], []
    for seed, fold in folds_by_seed.items():
        oof = oof_scores(make, columns, X, y, fold)
        results.append(summarize(keys, y, oof, fold))
        oofs.append(oof)
    all_folds = [v for r in results for v in r["fold_maa"]]
    return ({"mean_maa": float(np.mean(all_folds)), "std_fold_maa": float(np.std(all_folds)), "repeat_means": [r["mean_maa"] for r in results],
             "mean_first_accuracy": float(np.mean([r["mean_first_accuracy"] for r in results])), "roc_auc": float(np.mean([r["roc_auc"] for r in results])), "fold_maa": all_folds},
            np.mean(oofs, axis=0))
