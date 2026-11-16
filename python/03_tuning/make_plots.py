"""Step 3 plots. Reads results/development.json and writes two figures next to it.
Run from the project root:  python3 python/03_tuning/make_plots.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
dev = json.loads((OUT / "development.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for name, color in [("lightgbm", "#1f5f8b"), ("xgboost", "#a3501c"), ("catboost", "#4b8f5a")]:
    curve = dev["optuna"][name]["best_so_far"]
    ax.plot(range(len(curve)), curve, color=color, label=NAMES[name])
ax.set_xlabel("Optuna trial")
ax.set_ylabel("Best search MAA so far")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "optuna_history.png", dpi=140)
plt.close(fig)

labels = {"P": "Popularity", "H": "History", "PH": "Popularity + history", "PHS": "+ similarity", "PHA": "+ audio", "PHAS": "All groups"}
ab = dev["ablation"]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar([labels[a["groups"]] for a in ab], [a["mean_fold_maa"] for a in ab], yerr=[a["std_fold_maa"] for a in ab], capsize=3, color="#1f5f8b")
ax.axhline(dev["selected"]["copy_last"]["mean_fold_maa"], color="#a3501c", ls="--", lw=1)
ax.set_ylim(0.34, 0.62)
ax.set_ylabel("Development CV MAA (bars: fold SD)")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT / "ablation.png", dpi=140)
plt.close(fig)
print("wrote optuna_history.png and ablation.png")
