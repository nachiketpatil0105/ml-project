"""Step 5 plots. Reads results/error_analysis.json and writes by_position.png and calibration.png.
Run from the project root:  python3 python/05_error_analysis/make_plots.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
err = json.loads((OUT / "error_analysis.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
bp = err["by_position"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot([b["position"] for b in bp], [b["model"] for b in bp], marker="o", color="#1f5f8b", label="Boosted trees")
ax.plot([b["position"] for b in bp], [b["copy_last"] for b in bp], marker="o", color="#a3501c", label="Copy last skip")
ax.set_xlabel("Position in the prediction window")
ax.set_ylabel("Test accuracy")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "by_position.png", dpi=140)
plt.close(fig)
cal = err["calibration"]
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot([0, 1], [0, 1], color="#9aa5ac", ls="--")
ax.plot([c["mean_prob"] for c in cal], [c["observed"] for c in cal], marker="o", color="#1f5f8b")
ax.set_xlabel("Predicted skip probability")
ax.set_ylabel("Observed skip rate")
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=140)
print("wrote by_position.png and calibration.png")
