"""Model definitions, tuning grids and Optuna search spaces shared by the experiment and final-test scripts."""
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from features import GROUPS
from paths import QUICK

SEED = 42
JOBS = int(os.environ.get("SKIP_JOBS", os.cpu_count() or 1))
LR_CATEGORICAL = ["ctx_last", "rs_last", "re_last"]
LR_DROPPED = ["pat3", "pat4"]


def columns_for(groups):
    return sum((GROUPS[g] for g in groups), [])


def logistic(C, columns):
    """Scaling, imputation and one-hot encoding are pipeline steps, so they are fitted on training rows only."""
    categorical = [c for c in columns if c in LR_CATEGORICAL]
    numeric = [c for c in columns if c not in LR_CATEGORICAL and c not in LR_DROPPED]
    parts = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)]
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def random_forest(params):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=150, max_features="sqrt", n_jobs=JOBS, random_state=SEED, **params))])


def lightgbm(params):
    return lgb.LGBMClassifier(**{"subsample_freq": 1, "verbose": -1, "n_jobs": JOBS, "random_state": SEED, **params})


def xgboost(params):
    return xgb.XGBClassifier(**{"tree_method": "hist", "n_jobs": JOBS, "random_state": SEED, **params})


def catboost(params):
    return cb.CatBoostClassifier(**{"verbose": 0, "thread_count": JOBS, "random_seed": SEED, "allow_writing_files": False, **params})


def suggest_lightgbm(t):
    return {"n_estimators": t.suggest_int("n_estimators", 150, 800, step=50), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 4, 48), "min_child_samples": t.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 0.01, 50.0, log=True), "reg_alpha": t.suggest_float("reg_alpha", 0.001, 10.0, log=True)}


def suggest_xgboost(t):
    return {"n_estimators": t.suggest_int("n_estimators", 150, 800, step=50), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 7), "min_child_weight": t.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 0.01, 50.0, log=True), "reg_alpha": t.suggest_float("reg_alpha", 0.001, 10.0, log=True), "gamma": t.suggest_float("gamma", 0.0, 5.0)}


def suggest_catboost(t):
    return {"iterations": t.suggest_int("iterations", 200, 800, step=100), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": t.suggest_int("depth", 3, 8), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": t.suggest_float("random_strength", 0.1, 5.0, log=True), "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 3.0)}


# The first Optuna trial of each library is the configuration used before the search, so the search can only improve on it.
STARTING_POINTS = {
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 5.0, "reg_alpha": 0.001},
    "xgboost": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 5.0, "reg_alpha": 0.001, "gamma": 0.0},
    "catboost": {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
}
BOOSTERS = {"lightgbm": (lightgbm, suggest_lightgbm), "xgboost": (xgboost, suggest_xgboost), "catboost": (catboost, suggest_catboost)}
SEARCH_SPACES = {
    "lightgbm": "n_estimators 150-800, learning_rate 0.01-0.1 (log), num_leaves 4-48, min_child_samples 10-200 (log), subsample 0.5-1, colsample_bytree 0.3-1, reg_lambda 0.01-50 (log), reg_alpha 0.001-10 (log)",
    "xgboost": "n_estimators 150-800, learning_rate 0.01-0.1 (log), max_depth 2-7, min_child_weight 1-50 (log), subsample 0.5-1, colsample_bytree 0.3-1, reg_lambda 0.01-50 (log), reg_alpha 0.001-10 (log), gamma 0-5",
    "catboost": "iterations 200-800, learning_rate 0.02-0.15 (log), depth 3-8, l2_leaf_reg 1-30 (log), random_strength 0.1-5 (log), bagging_temperature 0-3",
}
RF_GRID = [{"max_depth": d, "min_samples_leaf": l} for d in (8, 12, 16) for l in (10, 30)]
LR_GRID = [0.01, 0.1, 1.0]


def make_model(name, params, columns=None):
    if name in BOOSTERS:
        return BOOSTERS[name][0](params)
    if name == "random_forest":
        return random_forest(params)
    if name in ("history_lr", "popularity_lr"):
        return logistic(params, columns)
    raise KeyError(name)


def complexity(name, params):
    """Smaller is simpler; used only to break near-ties in CV score."""
    if params is None:
        return (0,)
    if name == "lightgbm":
        return (params["num_leaves"] * params["n_estimators"],)
    if name == "xgboost":
        return (params["max_depth"] * params["n_estimators"],)
    if name == "catboost":
        return (params["depth"] * params["iterations"],)
    if name == "random_forest":
        return (params["max_depth"], -params["min_samples_leaf"])
    return (params,)
