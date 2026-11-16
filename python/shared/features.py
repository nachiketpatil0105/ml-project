"""Session loading, the saved session split, and feature construction.

A session of L tracks has K = ceil(L/2) observed tracks (the history) and L - K tracks to predict.

Every feature for a prediction row uses only (a) the first half of the same
session, (b) the metadata of the candidate track and (c) the queue position.
Labels of the prediction window are returned separately and never enter X.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from paths import ROOT, results
VECTORS = [f"acoustic_vector_{i}" for i in range(8)]
TRACK_COLUMNS = ["duration", "release_year", "us_popularity_estimate", "acousticness", "beat_strength", "bounciness", "danceability", "dyn_range_mean", "energy", "flatness", "instrumentalness", "key", "liveness", "loudness", "mechanism", "mode", "organism", "speechiness", "tempo", "time_signature", "valence"] + VECTORS
CONTEXTS = ["user_collection", "editorial_playlist", "catalog", "radio", "personalized_playlist", "charts"]
REASON_START = ["trackdone", "fwdbtn", "clickrow", "backbtn", "appload", "playbtn", "remote", "endplay", "trackerror"]
REASON_END = ["trackdone", "fwdbtn", "backbtn", "endplay", "logout", "clickrow", "remote", "appload"]
PAUSE_COLUMNS = ["no_pause_before_play", "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback"]
RADAR = ["energy", "danceability", "valence", "acousticness", "speechiness"]

HISTORY_COLUMNS = (["L", "k", "pos", "relpos", "abspos", "h_skip2", "last1", "last2", "last3", "last4", "last5", "last3_rate", "last5_rate", "streak_skip", "streak_keep", "pat3", "pat4", "hour", "dow", "premium", "shuffle", "ctx_last", "ctx_switch_mean", "uniq_ratio", "repeat", "rs_last", "re_last"]
    + ["ctx_" + c for c in CONTEXTS] + ["rs_" + c for c in REASON_START] + ["re_" + c for c in REASON_END] + ["m_" + c for c in PAUSE_COLUMNS])
POPULARITY = ["t_us_popularity_estimate"]
ACOUSTIC = ["t_" + c for c in TRACK_COLUMNS if c != "us_popularity_estimate"]
SIMILARITY = ["cos_keep", "euc_keep", "cos_skip", "euc_skip", "d_keep_minus_skip", "euc_prevtrack", "euc_lastkept"]
GROUPS = {"P": POPULARITY, "H": HISTORY_COLUMNS, "A": ACOUSTIC, "S": SIMILARITY}
FEATURE_COLUMNS = POPULARITY + HISTORY_COLUMNS + ACOUSTIC + SIMILARITY
# Columns that hold category codes rather than ordered quantities.
CATEGORICAL_CODES = ["ctx_last", "rs_last", "re_last", "pat3", "pat4"]


def to_bool(series):
    values = series.astype("string").str.lower().str.strip()
    if (values.notna() & ~values.isin(["true", "false"])).any():
        raise ValueError("Unexpected boolean encoding in " + str(series.name))
    return (values == "true")


def load_data(data_dir=ROOT / "data"):
    log = pd.read_csv(Path(data_dir) / "log_mini.csv")
    tracks = pd.read_csv(Path(data_dir) / "tf_mini.csv")
    for column in ["skip_1", "skip_2", "skip_3"]:
        log[column] = to_bool(log[column])
    tracks["mode"] = (tracks["mode"] == "major").astype(float)
    return log.sort_values(["session_id", "session_position"]).reset_index(drop=True), tracks


def validate(log, tracks):
    if log[["session_id", "session_position", "session_length", "track_id_clean"]].isna().any().any():
        raise ValueError("Missing session keys")
    if log.duplicated(["session_id", "session_position"]).any():
        raise ValueError("Duplicate session positions")
    if tracks.track_id.duplicated().any() or not log.track_id_clean.isin(tracks.track_id).all():
        raise ValueError("Track metadata missing or duplicated")
    g = log.groupby("session_id").agg(n=("session_position", "size"), length=("session_length", "first"), last=("session_position", "max"))
    if not ((g.n == g.length) & (g["last"] == g.length) & g.length.between(10, 20)).all():
        raise ValueError("Incomplete sessions")


def session_split(session_ids, path=None, seed=42, test_size=0.2):
    """Create the 80/20 session split once; reuse it afterwards. Folds for cross-validation are built later, on development sessions only."""
    ids = sorted(set(session_ids))
    path = Path(path) if path else results("prep") / "session_split.json"
    settings = {"seed": seed, "test_size": test_size}
    if path.exists():
        split = json.loads(path.read_text())
        if split["settings"] != settings or sorted(split["development"] + split["test"]) != ids:
            raise ValueError("Saved split does not match the data or settings")
        return split
    dev, test = train_test_split(ids, test_size=test_size, random_state=seed)
    split = {"settings": settings, "development": sorted(dev), "test": sorted(test)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split


def first_half_size(length):
    """Observed history length. Odd-length sessions round up, following the challenge description."""
    return (length + 1) // 2


def trailing_run(flags, value):
    count = 0
    for flag in flags[::-1]:
        if flag != value:
            break
        count += 1
    return count


def history_summary(flags):
    """Features that depend only on the observed skip_2 flags of the first half."""
    k = len(flags)
    out = {"h_skip2": float(np.mean(flags)), "last3_rate": float(np.mean(flags[-3:])), "last5_rate": float(np.mean(flags[-5:])),
           "streak_skip": trailing_run(flags, 1), "streak_keep": trailing_run(flags, 0),
           "pat3": int("".join(str(int(v)) for v in flags[-3:]), 2), "pat4": int("".join(str(int(v)) for v in flags[-4:]), 2)}
    for i in range(1, 6):
        out["last%d" % i] = int(flags[-i]) if k >= i else -1
    return out


def similarity(candidate, history_vectors, flags, previous_vector):
    """Distances from a candidate to the kept / skipped history profiles."""
    flags = np.asarray(flags, dtype=bool)
    keep, skipped = history_vectors[~flags], history_vectors[flags]
    keep = keep[np.isfinite(keep).all(axis=1)]
    skipped = skipped[np.isfinite(skipped).all(axis=1)]
    ok = bool(np.isfinite(candidate).all())
    out = {}
    for name, group in [("keep", keep), ("skip", skipped)]:
        if ok and len(group):
            centre = group.mean(axis=0)
            out["euc_" + name] = float(np.linalg.norm(candidate - centre))
            out["cos_" + name] = float(1 - candidate @ centre / (np.linalg.norm(candidate) * np.linalg.norm(centre) + 1e-12))
        else:
            out["euc_" + name] = out["cos_" + name] = np.nan
    out["d_keep_minus_skip"] = out["euc_keep"] - out["euc_skip"]
    out["euc_lastkept"] = float(np.linalg.norm(candidate - keep[-1])) if ok and len(keep) else np.nan
    out["euc_prevtrack"] = float(np.linalg.norm(candidate - previous_vector)) if ok and np.isfinite(previous_vector).all() else np.nan
    return out


def build_features(log, tracks, session_ids=None):
    """Return (X, keys, y). One row per track in the second half of a session."""
    validate(log, tracks)
    if session_ids is not None:
        log = log[log.session_id.isin(set(session_ids))]
    meta = tracks.set_index("track_id")
    rows, keys = [], []
    for sid, s in log.groupby("session_id", sort=True):
        L = len(s)
        k = first_half_size(L)
        h, f = s.iloc[:k], s.iloc[k:]
        flags = h.skip_2.to_numpy().astype(int)
        base = {"L": L, "k": k, **history_summary(flags)}
        last = h.iloc[-1]
        base["hour"] = int(last.hour_of_day)
        base["dow"] = pd.Timestamp(last.date).dayofweek
        base["premium"] = int(str(last.premium).lower() == "true")
        base["shuffle"] = int(str(last.hist_user_behavior_is_shuffle).lower() == "true")
        base["ctx_last"] = CONTEXTS.index(last.context_type) if last.context_type in CONTEXTS else -1
        for c in CONTEXTS:
            base["ctx_" + c] = float((h.context_type == c).mean())
        base["ctx_switch_mean"] = float(h.context_switch.mean())
        for c in REASON_START:
            base["rs_" + c] = float((h.hist_user_behavior_reason_start == c).mean())
        for c in REASON_END:
            base["re_" + c] = float((h.hist_user_behavior_reason_end == c).mean())
        base["rs_last"] = REASON_START.index(last.hist_user_behavior_reason_start) if last.hist_user_behavior_reason_start in REASON_START else -1
        base["re_last"] = REASON_END.index(last.hist_user_behavior_reason_end) if last.hist_user_behavior_reason_end in REASON_END else -1
        for c in PAUSE_COLUMNS:
            base["m_" + c] = float(h[c].mean())
        ids = s.track_id_clean.to_numpy()
        base["uniq_ratio"] = len(set(ids)) / L
        history_vectors = meta.loc[h.track_id_clean, VECTORS].to_numpy(float)
        history_ids = set(h.track_id_clean)
        future_vectors = meta.loc[f.track_id_clean, VECTORS].to_numpy(float)
        for j, row in enumerate(f.itertuples(index=False), 1):
            d = dict(base, pos=j, relpos=j / (L - k), abspos=k + j)
            track = meta.loc[row.track_id_clean]
            for c in TRACK_COLUMNS:
                d["t_" + c] = float(track[c])
            previous = future_vectors[j - 2] if j > 1 else history_vectors[-1]
            d.update(similarity(future_vectors[j - 1], history_vectors, flags, previous))
            d["repeat"] = int(row.track_id_clean in history_ids) + int((ids == row.track_id_clean).sum() > 1)
            rows.append(d)
            keys.append({"session_id": sid, "session_position": int(row.session_position), "track_id_clean": row.track_id_clean, "y": int(row.skip_2)})
    X = pd.DataFrame(rows)[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    K = pd.DataFrame(keys)
    return X, K[["session_id", "session_position", "track_id_clean"]], K["y"].astype(int)
