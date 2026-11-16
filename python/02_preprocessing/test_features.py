"""Run from the project root: python3 -m unittest discover -s python/02_preprocessing -v"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import features as F
from metrics import session_aa


def tiny_log(skips, length=10):
    rows = []
    for i in range(length):
        rows.append({"session_id": "s", "session_position": i + 1, "session_length": length, "track_id_clean": "t%d" % (i % 4), "skip_1": False, "skip_2": bool(skips[i]), "skip_3": bool(skips[i]),
                     "not_skipped": not skips[i], "context_switch": 0, "no_pause_before_play": 0, "short_pause_before_play": 0, "long_pause_before_play": 0, "hist_user_behavior_n_seekfwd": 0,
                     "hist_user_behavior_n_seekback": 0, "hist_user_behavior_is_shuffle": "true", "hour_of_day": 12, "date": "2018-07-15", "premium": "true", "context_type": "catalog",
                     "hist_user_behavior_reason_start": "trackdone", "hist_user_behavior_reason_end": "trackdone"})
    return pd.DataFrame(rows)


def tiny_tracks():
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"track_id": ["t0", "t1", "t2", "t3"]})
    for c in F.TRACK_COLUMNS:
        t[c] = rng.random(4)
    t["mode"] = 1.0
    return t


class FeatureTests(unittest.TestCase):
    def test_second_half_labels_do_not_change_features(self):
        a, b = tiny_log([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]), tiny_log([1, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        Xa, _, ya = F.build_features(a, tiny_tracks())
        Xb, _, yb = F.build_features(b, tiny_tracks())
        pd.testing.assert_frame_equal(Xa, Xb)
        self.assertNotEqual(ya.tolist(), yb.tolist())

    def test_history_summary(self):
        s = F.history_summary(np.array([0, 1, 1, 0, 1]))
        self.assertEqual((s["last1"], s["streak_skip"], s["pat3"], s["h_skip2"]), (1, 1, 5, 0.6))

    def test_rows_cover_second_half_only(self):
        X, K, y = F.build_features(tiny_log([0] * 10), tiny_tracks())
        self.assertEqual(K.session_position.tolist(), [6, 7, 8, 9, 10])

    def test_odd_length_rounds_history_up(self):
        X, K, y = F.build_features(tiny_log([0] * 11, 11), tiny_tracks())
        self.assertEqual(K.session_position.tolist(), [7, 8, 9, 10, 11])
        self.assertEqual(int(X.k.iloc[0]), 6)

    def test_average_accuracy(self):
        self.assertAlmostEqual(session_aa([1, 1, 1, 0, 0]), 0.6)
        self.assertAlmostEqual(session_aa([0, 0, 1, 1, 1]), (1 / 3 * 1 + 2 / 4 * 1 + 3 / 5 * 1) / 5)


if __name__ == "__main__":
    unittest.main()
