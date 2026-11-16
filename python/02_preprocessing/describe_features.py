"""Step 2b. Build the feature table and describe it for the website.

Run from the project root:  python3 python/02_preprocessing/describe_features.py
Outputs in results/:
  feature_summary.json   feature groups and the list of allowed input columns, row counts,
                         share of missing values, and one worked example row
  features_preview.csv   the first 300 prediction rows of the development sessions (features + target)
Nothing here fits a model or looks at the test sessions' labels for any decision.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from features import load_data, session_split, build_features, GROUPS, FEATURE_COLUMNS, SIMILARITY, first_half_size
from paths import results

GROUP_NAMES = {"P": "Popularity", "H": "Listening history and queue position", "A": "Track audio and metadata", "S": "Similarity to the session's own history"}
GROUP_TEXT = {
    "P": "The popularity estimate of the upcoming track.",
    "H": "How the listener behaved in the first half (skip rate, last five skip flags, skip streaks, how tracks started and ended, pauses, seeks, context, premium, shuffle, hour), plus the session length and the position of the track in the queue.",
    "A": "Numeric audio descriptors and metadata of the upcoming track: duration, release year, acousticness, danceability, energy, loudness, tempo, valence and the eight acoustic vector values, among others.",
    "S": "How far the upcoming track is, in audio terms, from the tracks the listener kept, from the tracks the listener skipped, from the last kept track and from the previous track in the queue.",
}
EXAMPLE_COLUMNS = ["L", "k", "pos", "relpos", "h_skip2", "last1", "last2", "last3", "streak_skip", "re_trackdone", "t_us_popularity_estimate", "t_energy", "t_danceability", "t_valence", "t_tempo", "euc_keep", "euc_skip"]


def main():
    log, tracks = load_data()
    split = session_split(log.session_id)
    X, keys, y = build_features(log, tracks)
    dev = keys.session_id.isin(split["development"]).to_numpy()
    # one worked example: the first development session of length 12
    lengths = log.groupby("session_id").session_length.first()
    sid = next(s for s in split["development"] if lengths[s] == 12)
    rows = np.flatnonzero((keys.session_id == sid).to_numpy())
    first = rows[0]
    history = log[log.session_id == sid].sort_values("session_position").skip_2.astype(int).tolist()[:first_half_size(12)]
    example = {"session_length": 12, "history_skip_2": history, "row_position": int(keys.session_position[first]),
               "features": {c: (None if not np.isfinite(X[c][first]) else float(X[c][first])) for c in EXAMPLE_COLUMNS}, "target_skip_2": int(y[first])}
    summary = {"first_half_rule": "ceil(L/2)", "feature_count": len(FEATURE_COLUMNS),
               "sessions": {"development": len(split["development"]), "test": len(split["test"]), "total": len(split["development"]) + len(split["test"])},
               "rows": {"development": int(dev.sum()), "test": int((~dev).sum()), "total": int(len(X))},
               "target_skip_rate": {"development": float(y[dev].mean()), "test": float(y[~dev].mean())},
               "groups": {g: {"name": GROUP_NAMES[g], "description": GROUP_TEXT[g], "count": len(cols), "columns": cols} for g, cols in GROUPS.items()},
               "missing_share": {c: float(X[c].isna().mean()) for c in SIMILARITY}, "example": example}
    out = results("prep")
    (out / "feature_summary.json").write_text(json.dumps(summary, indent=1))
    preview = X[dev].head(300).copy()
    preview.insert(0, "session_position", keys.session_position[dev].head(300).to_numpy())
    preview["target_skip_2"] = y[dev].head(300).to_numpy()
    preview.to_csv(out / "features_preview.csv", index=False)
    print("rows development/test:", summary["rows"]["development"], summary["rows"]["test"], "features:", len(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
