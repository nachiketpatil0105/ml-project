"""Step 3. Tuning and model comparison on development sessions only.

Uses the 8,000 development sessions. Hyperparameters are searched with Optuna (TPE) using
stratified, session-grouped 5-fold CV (search seed). The best trials are then re-evaluated with
three further repeats of 5-fold CV (different seeds) and the final choice is made on that
repeated estimate. The test sessions are never loaded into any fit or comparison.

Run from the project root:   python3 python/03_tuning/run_experiments.py [--trials 40] [--top 5] [--force]
Outputs in results/: development.json (everything the website shows), frozen_config.json (the choices made, read by step 4),
cv_folds.json (exact fold assignments), development_oof_primary.csv, stages/ and optuna.db (cache; an interrupted run resumes).
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models import *
from cv import SEARCH_SEED, REPEAT_SEEDS, fold_assignment, oof_scores, summarize, repeated_cv
from metrics import per_session
from features import load_data, session_split, build_features, FEATURE_COLUMNS
from paths import results, QUICK

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
TIE = 0.001


def aggregate(results):
    all_folds = [v for r in results for v in r["fold_maa"]]
    return {"mean_maa": float(np.mean(all_folds)), "std_fold_maa": float(np.std(all_folds)), "repeat_means": [r["mean_maa"] for r in results],
            "mean_first_accuracy": float(np.mean([r["mean_first_accuracy"] for r in results])), "roc_auc": float(np.mean([r["roc_auc"] for r in results])), "fold_maa": all_folds}


def entry(params, r):
    return {"params": params, "mean_fold_maa": r["mean_maa"], "std_fold_maa": r["std_fold_maa"], "repeat_means": r["repeat_means"],
            "mean_fold_first_accuracy": r["mean_first_accuracy"], "roc_auc": r["roc_auc"], "fold_maa": r["fold_maa"]}


def choose(candidates, name):
    best = max(c["score"] for c in candidates)
    tied = [c for c in candidates if best - c["score"] <= TIE]
    return min(tied, key=lambda c: complexity(name, c["params"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3 if QUICK else 40)
    parser.add_argument("--top", type=int, default=2 if QUICK else 5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()
    log, tracks = load_data()
    split = session_split(log.session_id)
    X, keys, y = build_features(log, tracks, split["development"])
    if QUICK:
        keep = keys.session_id.isin(sorted(keys.session_id.unique())[:1000]).to_numpy()
        X, keys, y = X[keep].reset_index(drop=True), keys[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    print("development rows", len(X), "sessions", keys.session_id.nunique(), flush=True)
    PROCESSED = results("tune")
    stages = PROCESSED / "stages"
    fingerprint = hashlib.md5(repr((len(X), tuple(FEATURE_COLUMNS), round(float(y.mean()), 6), args.trials, args.top)).encode()).hexdigest()
    mark = PROCESSED / "fingerprint.txt"
    if args.force or (mark.exists() and mark.read_text() != fingerprint):
        shutil.rmtree(stages, ignore_errors=True)
        (PROCESSED / "optuna.db").unlink(missing_ok=True)
    stages.mkdir(exist_ok=True)
    mark.write_text(fingerprint)

    def stage(name, fn):
        path, oof_path = stages / (name + ".json"), stages / (name + "_oof.npy")
        if path.exists():
            print("cached", name, flush=True)
            return json.loads(path.read_text()), (np.load(oof_path) if oof_path.exists() else None)
        result, oof = fn()
        path.write_text(json.dumps(result))
        if oof is not None:
            np.save(oof_path, oof)
        print("done", name, "%.0fs" % (time.time() - started), flush=True)
        return result, oof

    folds = {seed: fold_assignment(keys, y, seed) for seed in [SEARCH_SEED] + REPEAT_SEEDS}
    (PROCESSED / "cv_folds.json").write_text(json.dumps({str(s): dict(zip(keys.session_id.astype(str), map(int, f))) for s, f in folds.items()}))
    search_fold = folds[SEARCH_SEED]
    repeat_folds = {s: folds[s] for s in REPEAT_SEEDS}
    full = FEATURE_COLUMNS
    selected, grids, optuna_report, oofs = {}, {}, {}, {}

    # Heuristic baselines: no tuning, evaluated with the same repeated folds.
    def rules():
        funcs = {"global_majority": lambda fold, f: np.full(int((fold == f).sum()), float(y[fold != f].mean() > 0.5)), "history_majority": lambda fold, f: (X.h_skip2[fold == f] > 0.5).astype(float).to_numpy(),
                 "copy_last": lambda fold, f: X.last1[fold == f].astype(float).to_numpy()}
        out = {}
        for name, fn in funcs.items():
            results = []
            for seed, fold in repeat_folds.items():
                oof = np.zeros(len(X))
                for f in range(5):
                    oof[fold == f] = fn(fold, f)
                results.append(summarize(keys, y, oof, fold))
            out[name] = entry(None, aggregate(results))
        return out, None
    base, _ = stage("baselines", rules)
    selected.update(base)

    def tuned_grid(name, grid, columns):
        def run():
            rows = []
            for p in (grid[:1] if QUICK else grid):
                oof = oof_scores(lambda: make_model(name, p, columns), columns, X, y, search_fold)
                s = summarize(keys, y, oof, search_fold)
                rows.append({"params": p, "score": s["mean_maa"], "search": s})
                print(name, p, round(s["mean_maa"], 4), flush=True)
            pick = choose(rows, name)
            r, oof = repeated_cv(lambda: make_model(name, pick["params"], columns), columns, X, y, keys, repeat_folds)
            return {"grid": [{"params": c["params"], "mean_fold_maa": c["search"]["mean_maa"], "mean_fold_first_accuracy": c["search"]["mean_first_accuracy"], "roc_auc": c["search"]["roc_auc"]} for c in rows],
                    "chosen": entry(pick["params"], r), "index": rows.index(pick)}, oof
        return stage(name, run)

    for name, grid, columns in [("popularity_lr", LR_GRID, columns_for("P")), ("history_lr", LR_GRID, columns_for("PH")), ("random_forest", RF_GRID, full)]:
        res, oof = tuned_grid(name, grid, columns)
        grids[name] = {"grid": res["grid"], "index": res["index"]}
        selected[name] = res["chosen"]

    def booster(name):
        make, suggest = BOOSTERS[name]

        def run():
            study = optuna.create_study(study_name=name, storage="sqlite:///%s" % (PROCESSED / "optuna.db"), direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), load_if_exists=True)
            if not study.trials:
                study.enqueue_trial(STARTING_POINTS[name])

            def objective(trial):
                params = suggest(trial)
                s = summarize(keys, y, oof_scores(lambda: make(params), full, X, y, search_fold), search_fold)
                trial.set_user_attr("first_accuracy", s["mean_first_accuracy"])
                return s["mean_maa"]

            def report(study, trial):
                print(name, "trial", trial.number, round(trial.value, 4), "best", round(study.best_value, 4), "%.0fs" % (time.time() - started), flush=True)
            done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(done) < args.trials:
                study.optimize(objective, n_trials=args.trials - len(done), callbacks=[report])
            trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.number)
            best_so_far = np.maximum.accumulate([t.value for t in trials]).tolist()
            top = sorted(trials, key=lambda t: t.value, reverse=True)[:args.top]
            candidates, table = [], []
            for t in top:
                r, oof = repeated_cv(lambda: make(t.params), full, X, y, keys, repeat_folds)
                candidates.append({"params": t.params, "score": r["mean_maa"], "result": r, "oof": oof})
                table.append({"trial": t.number, "params": t.params, "search_maa": t.value, "repeated_maa": r["mean_maa"], "repeat_means": r["repeat_means"]})
                print(name, "re-evaluated trial", t.number, round(r["mean_maa"], 4), flush=True)
            pick = choose(candidates, name)
            info = {"trials": len(trials), "best_so_far": best_so_far, "starting_point_search_maa": trials[0].value, "best_search_maa": max(t.value for t in trials), "top": table,
                    "chosen_trial": table[candidates.index(pick)]["trial"], "search_space": SEARCH_SPACES[name]}
            return {"optuna": info, "chosen": entry(pick["params"], pick["result"])}, pick["oof"]
        return stage("optuna_" + name, run)

    for name in BOOSTERS:
        res, oof = booster(name)
        optuna_report[name] = res["optuna"]
        selected[name] = res["chosen"]
        oofs[name] = oof

    scores = {n: selected[n]["mean_fold_maa"] for n in BOOSTERS}
    primary = "lightgbm" if max(scores.values()) - scores["lightgbm"] <= TIE else max(scores, key=scores.get)
    oof = oofs[primary]
    grid = [round(t, 2) for t in np.arange(0.40, 0.601, 0.01)]
    curve = {t: float(per_session(keys, y, oof, t).aa.mean()) for t in grid}
    threshold = 0.5 if max(curve.values()) - curve[0.5] < TIE else max(curve, key=curve.get)

    def ablate():
        out = []
        for groups in ["P", "H", "PH", "PHA", "PHS", "PHAS"]:
            cols = columns_for(groups)
            r, _ = repeated_cv(lambda: make_model(primary, selected[primary]["params"]), cols, X, y, keys, repeat_folds)
            out.append({"groups": groups, "features": len(cols), "mean_fold_maa": r["mean_maa"], "std_fold_maa": r["std_fold_maa"], "mean_fold_first_accuracy": r["mean_first_accuracy"], "roc_auc": r["roc_auc"]})
            print("ablation", groups, round(r["mean_maa"], 4), flush=True)
        return {"rows": out}, None
    ablation = stage("ablation_" + primary, ablate)[0]["rows"]

    rule = ("Search: Optuna TPE, %d trials per boosting library, objective = mean fold MAA of stratified session-grouped 5-fold CV. "
            "Choice: the best %d trials are re-evaluated with 3 more repeats of 5-fold CV (15 folds); the highest mean MAA wins, configurations within 0.001 are tied and the smaller one is kept. "
            "Among libraries within 0.001, LightGBM is the primary model." % (args.trials, args.top))
    frozen = {"primary": primary, "threshold": threshold, "tie_tolerance": TIE, "selected": selected, "threshold_curve": curve, "selection_rule": rule}
    (PROCESSED / "frozen_config.json").write_text(json.dumps(frozen, indent=1))
    pd.DataFrame({"session_id": keys.session_id, "session_position": keys.session_position, "y": y, "score_repeat_mean": oof}).to_csv(PROCESSED / "development_oof_primary.csv", index=False)
    report = {"development_sessions": int(keys.session_id.nunique()), "development_rows": int(len(X)), "cv": {"method": "StratifiedGroupKFold, 5 folds, grouped by session", "search_seed": SEARCH_SEED, "repeat_seeds": REPEAT_SEEDS},
              "trials": args.trials, "top": args.top, "grids": grids, "optuna": optuna_report, "selected": selected, "primary": primary, "threshold": threshold, "threshold_curve": curve,
              "ablation": ablation, "selection_rule": rule, "seconds": time.time() - started}
    (PROCESSED / "development.json").write_text(json.dumps(report, indent=1))
    print("primary", primary, "threshold", threshold, "seconds", round(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
