"""Step 4 plot. Reads results/test_results.json and writes model_comparison.png next to it.
Run from the project root:  python3 python/04_final_test/make_plots.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
res = json.loads((OUT / "test_results.json").read_text())
rows = {r["id"]: r for r in res["rows"]}
order = ["global_majority", "history_majority", "copy_last", "popularity_lr", "history_lr", "random_forest", "xgboost", "catboost", "lightgbm"]
rs = [rows[i] for i in order]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.5, 4))
ys = list(range(len(rs)))
ax.barh(ys, [r["maa"] for r in rs], color=["#9aa5ac"] * 4 + ["#7aa6c2"] * 2 + ["#1f5f8b"] * 3, height=0.6)
ax.errorbar([r["maa"] for r in rs], ys, xerr=[[r["maa"] - r["ci"]["low"] for r in rs], [r["ci"]["high"] - r["maa"] for r in rs]], fmt="none", ecolor="#222", capsize=3)
ax.set_yticks(ys)
ax.set_yticklabels([r["name"] for r in rs])
ax.invert_yaxis()
ax.axvline(rows["copy_last"]["maa"], color="#a3501c", ls="--", lw=1)
ax.set_xlim(0.35, 0.65)
ax.set_xlabel("Test MAA (bars: 95% session-bootstrap interval)")
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=140)
print("wrote model_comparison.png")
