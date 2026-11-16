"""Step 5. Error analysis of the primary model on the test sessions.
Needs no model fitting: it reads the test scores saved by step 4.

Run from the project root:  python3 python/05_error_analysis/analyze_errors.py
Output: results/error_analysis.json  (confusion-free tables by position, history, context, premium;
        calibration; four representative sessions)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from features import load_data, session_split, build_features, CONTEXTS, first_half_size
from metrics import per_session
from paths import results


def main():
    log, tracks = load_data()
    split = session_split(log.session_id)
    frozen = json.loads((results("tune") / "frozen_config.json").read_text())
    primary, threshold = frozen["primary"], frozen["threshold"]
    pred_table = pd.read_csv(results("final") / "test_predictions.csv")
    Xt, kt, yt = build_features(log, tracks, sorted(set(pred_table.session_id)))
    frame = kt.merge(pred_table[["session_id", "session_position", "y", primary, "copy_last"]], on=["session_id", "session_position"], how="left")
    assert len(frame) == len(pred_table) and (frame.y.to_numpy() == yt.to_numpy()).all(), "predictions do not line up with features"
    frame["prob"] = frame.pop(primary)
    frame["pred"] = (frame.prob > threshold).astype(int)
    frame["pos"], frame["h"], frame["premium"], frame["ctx"], frame["k"] = Xt.pos, Xt.h_skip2, Xt.premium, Xt.ctx_last, Xt.k
    frame["ok"] = (frame.pred == frame.y).astype(int)
    frame["ok_copy"] = (frame.copy_last.astype(int) == frame.y).astype(int)
    by_position = [{"position": int(p), "n": int(len(g)), "model": float(g.ok.mean()), "copy_last": float(g.ok_copy.mean())} for p, g in frame.groupby("pos") if len(g) >= 100]
    bins = pd.cut(frame.h, [-0.01, 0.34, 0.67, 1.0], labels=["0-33%", "34-67%", "68-100%"])
    by_history = [{"bin": str(b), "n": int(len(g)), "skip_rate": float(g.y.mean()), "model": float(g.ok.mean()), "copy_last": float(g.ok_copy.mean())} for b, g in frame.groupby(bins, observed=True)]
    by_context = [{"context": CONTEXTS[int(c)] if c >= 0 else "unknown", "n": int(len(g)), "skip_rate": float(g.y.mean()), "model": float(g.ok.mean()), "copy_last": float(g.ok_copy.mean())} for c, g in frame.groupby("ctx") if len(g) >= 200]
    by_premium = [{"premium": bool(p), "n": int(len(g)), "skip_rate": float(g.y.mean()), "model": float(g.ok.mean()), "copy_last": float(g.ok_copy.mean())} for p, g in frame.groupby("premium")]
    cal_bins = pd.cut(frame.prob, np.linspace(0, 1, 11), include_lowest=True)
    calibration = [{"low": float(b.left), "high": float(b.right), "n": int(len(g)), "mean_prob": float(g.prob.mean()), "observed": float(g.y.mean())} for b, g in frame.groupby(cal_bins, observed=True) if len(g) >= 30]
    history_of = log.sort_values(["session_id", "session_position"]).groupby("session_id").skip_2.apply(lambda v: v.astype(int).tolist())

    def sequence(sid, title, why):
        g = frame[frame.session_id == sid].sort_values("session_position")
        k = first_half_size(len(history_of[sid]))
        return {"title": title, "why": why, "history": history_of[sid][:k], "prob": [round(float(v), 3) for v in g.prob], "truth": g.y.tolist(), "pred": g.pred.tolist()}
    wrong_skip = frame[(frame.y == 0) & (frame.pred == 1)].sort_values("prob", ascending=False).iloc[0]
    wrong_keep = frame[(frame.y == 1) & (frame.pred == 0)].sort_values("prob").iloc[0]
    sess = frame.groupby("session_id").agg(h=("h", "first"), skip=("y", "mean"), n=("y", "size"))
    momentum = sess[(sess.h == 1.0) & (sess.n >= 5)].sort_values("skip").index[0]
    table = per_session(kt, frame.y, frame.prob, threshold)
    typical = table.aa.sub(table.aa.median()).abs().sort_values(kind="stable").index[0]
    examples = [sequence(wrong_skip.session_id, "Most confident false skip", "The model gave this track its highest skip probability among tracks that were actually kept."),
                sequence(wrong_keep.session_id, "Most confident false keep", "The model gave this track its lowest skip probability among tracks that were actually skipped."),
                sequence(momentum, "Momentum broke", "Every first-half track was skipped, yet most of the second half was kept."),
                sequence(typical, "Typical session", "A session whose average accuracy is closest to the test median.")]
    out = {"model": primary, "threshold": threshold, "by_position": by_position, "by_history": by_history, "by_context": by_context, "by_premium": by_premium, "calibration": calibration, "examples": examples}
    (results("errors") / "error_analysis.json").write_text(json.dumps(out, indent=1))
    print("wrote error_analysis.json for", primary, "at threshold", threshold)


if __name__ == "__main__":
    main()
