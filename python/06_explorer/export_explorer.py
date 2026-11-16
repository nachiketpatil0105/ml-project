"""Step 6. Export what the website explorer needs.
Exports the trees of the LightGBM model that produced the test scores in step 4 (results/lightgbm_model.txt),
plus 24 fixed test sessions with their features and the model's predictions.
If that file is missing, the model is refitted with the parameters frozen in step 3 (results can then differ
slightly between machines, because LightGBM training is not bit-for-bit reproducible across setups).

Run from the project root:  python3 python/06_explorer/export_explorer.py
Outputs in results/: model.json (LightGBM trees), sessions.json (test sessions for the explorer)
"""
import json
import sys
import warnings
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from models import make_model
from features import load_data, session_split, build_features, first_half_size, FEATURE_COLUMNS, VECTORS, RADAR
from paths import results, QUICK

warnings.filterwarnings("ignore")


def compact_tree(node):
    if "leaf_value" in node:
        return {"v": round(node["leaf_value"], 6)}
    return {"f": node["split_feature"], "t": node["threshold"], "d": int(node["default_left"]), "m": node["missing_type"], "l": compact_tree(node["left_child"]), "r": compact_tree(node["right_child"])}


def main():
    log, tracks = load_data()
    split = session_split(log.session_id)
    frozen = json.loads((results("tune") / "frozen_config.json").read_text())
    if QUICK:
        split = {"development": split["development"][:1000], "test": split["test"][:300]}
    X, keys, y = build_features(log, tracks, split["development"] + split["test"])
    dev = keys.session_id.isin(split["development"]).to_numpy()
    saved = results("final") / "lightgbm_model.txt"
    if saved.exists():
        booster = lgb.Booster(model_file=str(saved))
        predict = lambda frame: booster.predict(frame)
    else:
        print("lightgbm_model.txt not found: refitting from the frozen parameters")
        model = make_model("lightgbm", frozen["selected"]["lightgbm"]["params"]).fit(X[dev][FEATURE_COLUMNS], y[dev])
        booster = model.booster_
        predict = lambda frame: model.predict_proba(frame)[:, 1]
    Xt, kt, yt = X[~dev].reset_index(drop=True), keys[~dev].reset_index(drop=True), y[~dev].reset_index(drop=True)
    dump = booster.dump_model()
    out = results("explorer")
    (out / "model.json").write_text(json.dumps({"features": FEATURE_COLUMNS, "trees": [compact_tree(t["tree_structure"]) for t in dump["tree_info"]], "threshold": frozen["threshold"], "params": frozen["selected"]["lightgbm"]["params"]}, separators=(",", ":")))
    rng = np.random.default_rng(42)
    ids = sorted(kt.session_id.unique())
    chosen = [ids[i] for i in rng.permutation(len(ids))[:60]]
    chosen = [s for s in chosen if Xt[kt.session_id == s].L.iloc[0] >= 14][:24]
    meta = tracks.set_index("track_id")
    enc = lambda a: [None if not np.isfinite(v) else float(v) for v in a]
    sessions = []
    for sid in chosen:
        mask = (kt.session_id == sid).to_numpy()
        s = log[log.session_id == sid]
        k = first_half_size(len(s))
        hist_ids = s.iloc[:k].track_id_clean
        feats = Xt.loc[mask, FEATURE_COLUMNS].to_numpy(float)
        probs = predict(Xt.loc[mask, FEATURE_COLUMNS])
        cand_ids = kt.loc[mask, "track_id_clean"]
        sessions.append({"label": "Session %d" % (len(sessions) + 1), "length": int(len(s)), "history": s.iloc[:k].skip_2.astype(int).tolist(),
                         "history_vectors": [enc(v) for v in meta.loc[hist_ids, VECTORS].to_numpy(float)], "history_radar": [enc(v) for v in meta.loc[hist_ids, RADAR].to_numpy(float)],
                         "rows": [{"features": enc(f), "candidate": enc(c), "radar": enc(r), "truth": int(t), "python_prob": round(float(p), 6)}
                                  for f, c, r, t, p in zip(feats, meta.loc[cand_ids, VECTORS].to_numpy(float), meta.loc[cand_ids, RADAR].to_numpy(float), yt[mask], probs)]})
    (out / "sessions.json").write_text(json.dumps({"features": FEATURE_COLUMNS, "radar": RADAR, "sessions": sessions}, separators=(",", ":")))
    print("exported", len(dump["tree_info"]), "trees and", len(sessions), "sessions")


if __name__ == "__main__":
    main()
