"""Step 7. "If we change one thing": small, scripted variations of the final pipeline.

Each experiment changes exactly one thing in the frozen main model (step 3) and re-scores it on the
development sessions with the same 15 repeated, session-grouped folds (seeds 101, 102, 103).
The test sessions are never loaded. This script replaces the hand-run exploratory notes of earlier
pipeline versions: every number it reports can be re-created from this repository.

Run from the project root:  python3 python/07_extra_experiments/run_extra.py
Needs: python/03_tuning/results/frozen_config.json and development.json (step 3).
Output: results/extra_experiments.json (read by the website, Team page and Preprocessing page).
"""
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import features as F
from features import load_data, session_split, build_features, FEATURE_COLUMNS, GROUPS
from models import make_model, columns_for
from cv import REPEAT_SEEDS, fold_assignment, oof_scores, summarize, repeated_cv
from paths import results, QUICK

warnings.filterwarnings("ignore")
AUDIO_DELTA = ["energy", "danceability", "valence", "tempo", "loudness", "acousticness"]
REASON_SHARES = ["rs_" + c for c in F.REASON_START] + ["re_" + c for c in F.REASON_END]
PRIOR_WEIGHT = 10  # smoothing of the per-song skip rate: (skips + 10 * prior) / (plays + 10)


def subset(X, keys, y, n):
    keep = keys.session_id.isin(sorted(keys.session_id.unique())[:n]).to_numpy()
    return X[keep].reset_index(drop=True), keys[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def first_half_extras(log, keys):
    """Per-session features from the first half (ceil rule) that the final model does NOT use: skip_1 / skip_3 history and mean audio."""
    log = log[log.session_id.isin(set(keys.session_id))]
    k = log.session_length.map(F.first_half_size)
    h = log[log.session_position <= k]
    g = h.groupby("session_id")
    out = pd.DataFrame({"h_skip1": g.skip_1.mean().astype(float), "h_skip3": g.skip_3.mean().astype(float),
                        "last_skip1": g.skip_1.last().astype(float), "last_skip3": g.skip_3.last().astype(float)})
    return out


def history_audio_means(log, tracks, keys):
    log = log[log.session_id.isin(set(keys.session_id))]
    h = log[log.session_position <= log.session_length.map(F.first_half_size)]
    h = h.merge(tracks[["track_id"] + AUDIO_DELTA], left_on="track_id_clean", right_on="track_id", how="left")
    return h.groupby("session_id")[AUDIO_DELTA].mean()


def fold_diff(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {"mean": float(d.mean()), "sd": float(d.std()), "folds_better": int((d > 0).sum()), "folds": int(len(d))}


def main():
    started = time.time()
    tune = results("tune")
    frozen = json.loads((tune / "frozen_config.json").read_text())
    dev_report = json.loads((tune / "development.json").read_text())
    primary = frozen["primary"]
    params = frozen["selected"][primary]["params"]
    log, tracks = load_data()
    split = session_split(log.session_id)
    X, keys, y = build_features(log, tracks, split["development"])
    if QUICK:
        X, keys, y = subset(X, keys, y, 600)
    folds = {s: fold_assignment(keys, y, s) for s in REPEAT_SEEDS}
    make = lambda: make_model(primary, params)
    experiments = []

    def run(name, columns, Xv=None, note=""):
        r, _ = repeated_cv(make, columns, X if Xv is None else Xv, y, keys, folds)
        print("%-55s MAA %.4f" % (name, r["mean_maa"]), "%.0fs" % (time.time() - started), flush=True)
        return r

    base = run("Final model (all four feature groups)", FEATURE_COLUMNS)

    def add(id_, change, kind, r, why, compare=True):
        row = {"id": id_, "change": change, "kind": kind, "mean_fold_maa": r["mean_maa"], "std_fold_maa": r["std_fold_maa"], "mean_fold_first_accuracy": r["mean_first_accuracy"],
               "roc_auc": r["roc_auc"], "delta_maa": r["mean_maa"] - base["mean_maa"], "why": why}
        if compare:
            row["paired"] = fold_diff(r["fold_maa"], base["fold_maa"])
        experiments.append(row)

    # 1. Remove one feature group at a time.
    for g, label in [("A", "audio and metadata"), ("S", "similarity"), ("P", "popularity")]:
        cols = [c for c in FEATURE_COLUMNS if c not in GROUPS[g]]
        add("drop_" + g, "Drop the %s features" % label, "remove", run("drop " + g, cols), "Tests whether group %s adds anything beyond the others." % g)
    # 2. Remove the start/end reason shares (part of the history group).
    cols = [c for c in FEATURE_COLUMNS if c not in REASON_SHARES]
    add("drop_reason_shares", "Drop the start/end reason shares (rs_*, re_*)", "remove", run("drop reason shares", cols), "These shares repeat the skip history in another form; this checks how much the model leans on them.")
    # 3. Add features we considered but did not keep.
    extra = first_half_extras(log, keys)
    Xs = X.copy()
    for c in extra.columns:
        Xs[c] = keys.session_id.map(extra[c]).to_numpy()
    add("add_skip1_skip3", "Add skip_1 / skip_3 history features", "add", run("add skip_1/skip_3", FEATURE_COLUMNS + list(extra.columns), Xs), "skip_1 and skip_3 are stricter and looser skip definitions; they are close copies of skip_2.")
    means = history_audio_means(log, tracks, keys)
    Xa = X.copy()
    for c in AUDIO_DELTA:
        Xa["delta_" + c] = Xa["t_" + c].to_numpy() - keys.session_id.map(means[c]).to_numpy()
    add("add_audio_deltas", "Add per-session audio deltas (song minus first-half mean)", "add", run("add audio deltas", FEATURE_COLUMNS + ["delta_" + c for c in AUDIO_DELTA], Xa), "Tests whether 'how different is this song from what was played' helps beyond the existing distances.")
    # 4. Per-song skip rate learned from other sessions, rebuilt inside every training fold (no leakage).
    def song_rate():
        dev_log = log[log.session_id.isin(set(keys.session_id))]
        results_, track = [], keys.track_id_clean.to_numpy()
        for seed, fold in folds.items():
            fold_of = dict(zip(keys.session_id, fold))
            lf = dev_log.session_id.map(fold_of).to_numpy()
            oof = np.zeros(len(X))
            for f in range(5):
                train_log = dev_log[lf != f]
                stats = train_log.groupby("track_id_clean").skip_2.agg(["sum", "count"])
                prior = float(train_log.skip_2.mean())
                own = train_log.groupby(["session_id", "track_id_clean"]).skip_2.agg(["sum", "count"])
                s = pd.Series(track).map(stats["sum"]).fillna(0).to_numpy()
                n = pd.Series(track).map(stats["count"]).fillna(0).to_numpy()
                tr = fold != f
                idx = pd.MultiIndex.from_arrays([keys.session_id[tr], keys.track_id_clean[tr]])
                s_tr = s[tr] - own["sum"].reindex(idx).fillna(0).to_numpy()  # leave the row's own session out
                n_tr = n[tr] - own["count"].reindex(idx).fillna(0).to_numpy()
                rate = np.empty(len(X))
                rate[tr] = (s_tr + PRIOR_WEIGHT * prior) / (n_tr + PRIOR_WEIGHT)
                rate[~tr] = (s[~tr] + PRIOR_WEIGHT * prior) / (n[~tr] + PRIOR_WEIGHT)
                Xr = X.copy()
                Xr["song_skip_rate"] = rate
                model = make().fit(Xr.loc[tr, FEATURE_COLUMNS + ["song_skip_rate"]], y[tr])
                oof[~tr] = model.predict_proba(Xr.loc[~tr, FEATURE_COLUMNS + ["song_skip_rate"]])[:, 1]
            results_.append(summarize(keys, y, oof, fold))
        all_folds = [v for r in results_ for v in r["fold_maa"]]
        r = {"mean_maa": float(np.mean(all_folds)), "std_fold_maa": float(np.std(all_folds)), "fold_maa": all_folds,
             "mean_first_accuracy": float(np.mean([r["mean_first_accuracy"] for r in results_])), "roc_auc": float(np.mean([r["roc_auc"] for r in results_]))}
        print("%-55s MAA %.4f" % ("add per-song skip rate", r["mean_maa"]), flush=True)
        return r
    add("add_song_skip_rate", "Add a per-song skip rate from other sessions", "add", song_rate(), "Learned inside each training fold, leaving the row's own session out. About 7 in 10 songs appear in only one session, so for many rows the rate is close to the overall skip rate.")
    # 5. Change the model, keep the features.
    lr = frozen["selected"]["history_lr"]
    r, _ = repeated_cv(lambda: make_model("history_lr", lr["params"], columns_for("PH")), columns_for("PH"), X, y, keys, folds)
    add("model_logistic", "Replace LightGBM by logistic regression (popularity + history)", "model", r, "Same folds; shows how much the non-linear model adds.")
    # 6. Change the decision threshold (no refit; read from step 3's threshold curve on the same out-of-fold scores).
    curve = {float(k): v for k, v in dev_report["threshold_curve"].items()}
    best_t = max(curve, key=curve.get)
    threshold = {"chosen": frozen["threshold"], "best": best_t, "maa_at_0.5": curve[0.5], "maa_at_best": curve[best_t], "gain": curve[best_t] - curve[0.5],
                 "rule": "move away from 0.5 only if the gain is at least 0.001", "curve": curve}
    # 7. Change the first-half rule: floor(L/2) instead of ceil(L/2). This changes the rows being predicted, so it is not a paired comparison.
    original = F.first_half_size
    F.first_half_size = lambda L: L // 2
    try:
        Xf, kf, yf = build_features(log, tracks, split["development"])
    finally:
        F.first_half_size = original
    if QUICK:
        Xf, kf, yf = subset(Xf, kf, yf, 600)
    ff = {s: fold_assignment(kf, yf, s) for s in REPEAT_SEEDS}
    r, _ = repeated_cv(make, FEATURE_COLUMNS, Xf, yf, kf, ff)
    print("%-55s MAA %.4f" % ("first half = floor(L/2)", r["mean_maa"]), flush=True)
    add("rule_floor", "First half = floor(L/2) instead of ceil(L/2)", "rule", r, "Odd-length sessions get one more song to predict and one fewer history song. The rows differ, so this is not a paired comparison.", compare=False)
    experiments[-1]["rows"] = int(len(Xf))
    out = {"primary": primary, "params": params, "base": {"mean_fold_maa": base["mean_maa"], "std_fold_maa": base["std_fold_maa"], "mean_fold_first_accuracy": base["mean_first_accuracy"], "roc_auc": base["roc_auc"], "rows": int(len(X))},
           "folds": "5-fold StratifiedGroupKFold by session, seeds %s (15 folds)" % REPEAT_SEEDS, "experiments": experiments, "threshold": threshold,
           "notes": "delta_maa = variant minus final model. paired.folds_better = number of the 15 folds where the variant scored higher. Development sessions only.",
           "seconds": time.time() - started}
    (results("extra") / "extra_experiments.json").write_text(json.dumps(out, indent=1))
    print("saved", results("extra") / "extra_experiments.json")


if __name__ == "__main__":
    main()
