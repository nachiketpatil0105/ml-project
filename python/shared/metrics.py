"""Mean Average Accuracy (MAA) and the other reported metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def session_aa(correct):
    """AA = (1/T) * sum_i A(i) * L(i), with A(i) the accuracy of the first i predictions."""
    correct = np.asarray(correct, dtype=float)
    return float(np.mean(correct * np.cumsum(correct) / np.arange(1, len(correct) + 1)))


def per_session(keys, y, prob, threshold=0.5):
    """Return one row per session: AA and whether the first prediction was right."""
    frame = keys[["session_id", "session_position"]].copy()
    frame["correct"] = ((np.asarray(prob) > threshold).astype(int) == np.asarray(y)).astype(int)
    frame = frame.sort_values(["session_id", "session_position"])
    grouped = frame.groupby("session_id", sort=True).correct
    return pd.DataFrame({"aa": grouped.apply(lambda v: session_aa(v.to_numpy())), "first": grouped.first()})


def evaluate(keys, y, prob, threshold=0.5):
    table = per_session(keys, y, prob, threshold)
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = (prob > threshold).astype(int)
    return {"maa": float(table.aa.mean()), "first_accuracy": float(table["first"].mean()),
            "accuracy": float((pred == y).mean()), "roc_auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None,
            "confusion": [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)],
            "sessions": int(len(table)), "rows": int(len(y))}


def paired_bootstrap(table_a, table_b, draws=2000, seed=42):
    """Resample sessions; return mean difference in MAA (a - b) and a 95% interval."""
    diff = (table_a.aa - table_b.aa.reindex(table_a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(draws)])
    return {"difference": float(diff.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def bootstrap_interval(table, draws=2000, seed=42):
    values = table.aa.to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)])
    return {"low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}
