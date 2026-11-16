"""Step 4. Final test. Reads the choices frozen by step 3, refits every model on all development
sessions, and scores the 2,000 test sessions once.

Run from the project root:  python3 python/04_final_test/run_final.py
Outputs in results/:
  test_results.json      test metrics of every model, 95% session-bootstrap intervals, paired
                         comparisons, feature-group importance (read by the website)
  test_predictions.csv   the score of every model for every test row (used by step 5)
  lightgbm_model.txt     the fitted LightGBM model that produced the test scores (used by step 6)
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models import make_model, columns_for
from metrics import evaluate, per_session, paired_bootstrap, bootstrap_interval
from features import load_data, session_split, build_features, FEATURE_COLUMNS, GROUPS
from paths import results, QUICK

warnings.filterwarnings("ignore")
NAMES = {"global_majority": "Training-label majority", "history_majority": "First-half skip majority", "copy_last": "Copy last skip", "popularity_lr": "Popularity-only logistic regression",
         "history_lr": "Logistic regression (popularity + history)", "random_forest": "Random forest", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}


def fit_model(name, params, X, y):
    cols = columns_for("P") if name == "popularity_lr" else columns_for("PH") if name == "history_lr" else FEATURE_COLUMNS
    return make_model(name, params, cols).fit(X[cols], y), cols


def main():
    log, tracks = load_data()
    split = session_split(log.session_id)
    frozen = json.loads((results("tune") / "frozen_config.json").read_text())
    if QUICK:
        split = {"development": split["development"][:1000], "test": split["test"][:300]}
    X, keys, y = build_features(log, tracks, split["development"] + split["test"])
    dev = keys.session_id.isin(split["development"]).to_numpy()
    Xd, yd = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True)
    Xt, kt, yt = X[~dev].reset_index(drop=True), keys[~dev].reset_index(drop=True), y[~dev].reset_index(drop=True)
    print("test sessions", kt.session_id.nunique(), "rows", len(Xt), flush=True)
    scores, fitted = {}, {}
    scores["global_majority"] = np.full(len(Xt), float(yd.mean() > 0.5))
    scores["history_majority"] = (Xt.h_skip2 > 0.5).astype(float).to_numpy()
    scores["copy_last"] = Xt.last1.astype(float).to_numpy()
    for name in ["popularity_lr", "history_lr", "random_forest", "lightgbm", "xgboost", "catboost"]:
        model, cols = fit_model(name, frozen["selected"][name]["params"], Xd, yd)
        scores[name] = model.predict_proba(Xt[cols])[:, 1]
        fitted[name] = model
        print("scored", name, flush=True)
    primary, threshold = frozen["primary"], frozen["threshold"]
    rows, tables = [], {}
    for name, s in scores.items():
        tables[name] = per_session(kt, yt, s)
        rows.append({"id": name, "name": NAMES[name], **evaluate(kt, yt, s), "ci": bootstrap_interval(tables[name]),
                     "dev_mean_fold_maa": frozen["selected"][name]["mean_fold_maa"], "params": frozen["selected"][name]["params"]})
    tuned = evaluate(kt, yt, scores[primary], threshold)
    tuned_table = per_session(kt, yt, scores[primary], threshold)
    comparisons = {other: paired_bootstrap(tuned_table, tables[other]) for other in ["copy_last", "history_majority", "history_lr", "random_forest", "popularity_lr"]}
    importance = pd.Series(fitted[primary].feature_importances_, index=FEATURE_COLUMNS, dtype=float)
    if primary == "lightgbm":
        importance = pd.Series(fitted[primary].booster_.feature_importance("gain"), index=FEATURE_COLUMNS)
    importance = importance / importance.sum()
    out = {"test_sessions": int(kt.session_id.nunique()), "test_rows": int(len(kt)), "primary": primary, "threshold": threshold, "rows": rows,
           "primary_tuned": {**tuned, "ci": bootstrap_interval(tuned_table), "threshold": threshold}, "comparisons_vs_primary_tuned": comparisons,
           "group_importance": {g: float(importance[c].sum()) for g, c in GROUPS.items()},
           "top_features": [{"feature": n, "share": float(v)} for n, v in importance.sort_values(ascending=False).head(12).items()],
           "test_skip_rate": float(yt.mean()), "bootstrap_draws": 2000, "first_half_rule": "ceil(L/2)"}
    folder = results("final")
    (folder / "test_results.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"session_id": kt.session_id, "session_position": kt.session_position, "y": yt, **{n: s for n, s in scores.items()}}).to_csv(folder / "test_predictions.csv", index=False)
    fitted["lightgbm"].booster_.save_model(str(folder / "lightgbm_model.txt"))
    print("primary", primary, "tuned MAA", round(tuned["maa"], 4), flush=True)


if __name__ == "__main__":
    main()
