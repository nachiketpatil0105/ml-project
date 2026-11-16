"""Describe the mini dataset and export website-ready JSON and figures.

Run from ml-project: python3 python/01_data_understanding/analyze.py
This is descriptive analysis of the supplied files, not model evaluation.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "next-track-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GREEN, ORANGE, LIGHT, INK = "#24684e", "#b56b46", "#b8cbb0", "#172c2b"
BOOLS = ["skip_1", "skip_2", "skip_3", "not_skipped", "premium", "hist_user_behavior_is_shuffle"]
INDICATORS = ["context_switch", "no_pause_before_play", "short_pause_before_play", "long_pause_before_play"]
FEATURES = ["acousticness", "danceability", "energy", "instrumentalness", "speechiness", "valence", "tempo", "loudness"]
IMPORTANT = [
    ("session_id", "log", "Anonymous session key, not a user ID", "Split and group by session; not a predictor."),
    ("session_position", "log", "Track position within the recorded session", "Orders the observed and target windows."),
    ("session_length", "log", "Number of recorded tracks, capped at 20", "Supplied offline task information; eventual live length may be unknown."),
    ("track_id_clean", "log", "Anonymous track key", "Join to track_features.track_id; not a song title."),
    ("skip_2", "log", "Brief-play skip indicator", "Prediction target; only first-half values may be model inputs. Exact time thresholds are not encoded in these CSV files."),
    ("skip_1 / skip_3", "log", "Alternative skip indicators", "Different definitions; neither is interchangeable with skip_2."),
    ("not_skipped", "log", "Complete-playback indicator", "Not the complement of skip_2."),
    ("hist_user_behavior_reason_start", "log", "Reason the current track started", "Descriptive association only for future rows; model may use observed history."),
    ("hist_user_behavior_reason_end", "log", "Reason the current track ended", "Reveals outcome information; exclude future values."),
    ("context_type / context_switch", "log", "Playback source and whether context changed", "Context can change; use only first-half information for prediction."),
    ("pause flags / seek counts", "log", "Pauses before playback and seeks within a play", "Interaction fields, not safe future predictors despite hist_ names."),
    ("premium / hist_user_behavior_is_shuffle", "log", "Subscription and shuffle indicators", "Use observed values; shuffle does not imply uniform random ordering."),
    ("date / hour_of_day", "log", "Recorded date and hour", "Describe sample coverage; no cross-session user linkage."),
    ("track_id", "track_features", "Unique metadata key", "Many-to-one join from track_id_clean."),
    ("us_popularity_estimate", "track_features", "Supplied US popularity estimate", "Static reference signal; not assumed to be current Spotify API popularity."),
    ("duration", "track_features", "Track duration (interpreted in seconds)", "Candidate metadata; inspect extreme values."),
    ("danceability / energy / valence / acousticness / speechiness", "track_features", "Interpretable acoustic descriptors", "Candidate features and explanatory profile charts."),
    ("tempo / loudness", "track_features", "Tempo and loudness descriptors", "Conventional units are BPM and dB; zero/extreme values are not automatically missing."),
    ("acoustic_vector_0 ... acoustic_vector_7", "track_features", "Eight supplied signed acoustic embedding components", "Candidate features and history-to-candidate distances."),
]


def records(frame):
    """Pandas conversion maps missing values to JSON null and numpy scalars to native values."""
    return json.loads(frame.to_json(orient="records", double_precision=10))


def grouped_rate(frame, key, target="skip_2"):
    return frame.groupby(key, observed=True)[target].agg(observations="count", positives="sum", rate="mean").reset_index()


def schema_summary(frame):
    result = []
    for name in frame:
        s = frame[name]
        entry = {"name": name, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique_values": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            finite = pd.to_numeric(s).replace([np.inf, -np.inf], np.nan).dropna()
            entry["summary"] = json.loads(finite.describe().to_json())
        elif s.nunique() <= 30:
            entry["value_counts"] = [{"value": str(k), "count": int(v)} for k, v in s.value_counts(dropna=False).items()]
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    out_dir = Path(__file__).resolve().parent / "results"
    image_dir = out_dir / "images"
    json_path = out_dir / "data_understanding.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(args.data_dir / "log_mini.csv")
    tracks = pd.read_csv(args.data_dir / "tf_mini.csv")
    if log.empty or tracks.empty:
        raise ValueError("Both input tables must contain data.")
    # Explicit conversion prevents the nonempty string 'false' from becoming True.
    invalid_boolean_counts = {}
    for col in BOOLS:
        normalized = log[col].astype("string").str.lower().str.strip()
        invalid_boolean_counts[col] = int((normalized.notna() & ~normalized.isin(["true", "false"])).sum())
        log[col] = normalized.map({"true": True, "false": False}).astype("boolean")
    if any(invalid_boolean_counts.values()):
        raise ValueError(f"Unexpected boolean values: {invalid_boolean_counts}. Correct the input interpretation before plotting.")
    if log[["session_id", "session_position", "session_length", "track_id_clean"]].isna().any().any():
        raise ValueError("Missing structural keys prevent reliable session-level summaries.")
    if tracks.track_id.isna().any() or tracks.track_id.duplicated().any():
        raise ValueError("Metadata track keys must be nonmissing and unique for the many-to-one join.")
    log = log.sort_values(["session_id", "session_position"]).reset_index(drop=True)
    sessions = log.groupby("session_id", sort=False).agg(
        recorded_length=("session_length", "first"), rows=("session_position", "size"),
        unique_positions=("session_position", "nunique"), first_position=("session_position", "min"),
        last_position=("session_position", "max"), length_values=("session_length", "nunique"))
    invalid_sessions = (~((sessions.rows == sessions.recorded_length) & (sessions.unique_positions == sessions.recorded_length)
        & (sessions.first_position == 1) & (sessions.last_position == sessions.recorded_length)
        & (sessions.length_values == 1) & sessions.recorded_length.between(10, 20))).sum()
    if invalid_sessions:
        raise ValueError(f"{invalid_sessions} sessions are incomplete or inconsistent; do not generate misleading session plots.")
    # Validate join cardinality without generating a play-weighted acoustic-feature table.
    joined = log[["track_id_clean"]].merge(tracks[["track_id"]], left_on="track_id_clean", right_on="track_id", how="left", validate="many_to_one", indicator=True)
    history = log.loc[log.session_position <= (log.session_length + 1) // 2]
    plays = log.groupby("track_id_clean").agg(plays=("skip_2", "size"), labelled_plays=("skip_2", "count"), skip_rate=("skip_2", "mean"))
    parsed_dates = pd.to_datetime(log.date, format="%Y-%m-%d", errors="coerce")
    length_counts = sessions.recorded_length.value_counts().sort_index()
    rate_length = grouped_rate(log, "session_length")
    rate_position = grouped_rate(log, "session_position")
    reason_rates = grouped_rate(log, "hist_user_behavior_reason_start").sort_values("rate")
    context_rates = grouped_rate(log, "context_type").sort_values("observations")
    # Source context means previous track's context, never the current destination.
    previous_context = log.groupby("session_id", sort=False).context_type.shift(1)
    context_source = log.assign(source_context=previous_context).dropna(subset=["source_context"])
    switch_position = grouped_rate(log, "session_position", "context_switch")
    switch_source = grouped_rate(context_source, "source_context", "context_switch").sort_values("rate")
    switch_length = grouped_rate(log, "session_length", "context_switch")
    numeric_nonfinite = {name: {c: int((~np.isfinite(df[c].dropna())).sum()) for c in df.select_dtypes(include="number")}
        for name, df in [("log", log), ("track_features", tracks)]}
    checks = {
        "missing_by_column": {"log": {c: int(n) for c, n in log.isna().sum().items()}, "track_features": {c: int(n) for c, n in tracks.isna().sum().items()}},
        "duplicate_log_rows": int(log.duplicated().sum()), "duplicate_track_rows": int(tracks.duplicated().sum()),
        "duplicate_session_positions": int(log.duplicated(["session_id", "session_position"]).sum()),
        "duplicate_track_ids": int(tracks.track_id.duplicated().sum()), "invalid_sessions": int(invalid_sessions),
        "unmatched_log_tracks_rows": int((joined._merge == "left_only").sum()),
        "unused_metadata_tracks": int((~tracks.track_id.isin(log.track_id_clean)).sum()),
        "invalid_boolean_counts": invalid_boolean_counts,
        "invalid_indicator_counts": {c: int((log[c].notna() & ~log[c].isin([0, 1])).sum()) for c in INDICATORS},
        "negative_seek_counts": {c: int((log[c] < 0).sum()) for c in ["hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback"]},
        "invalid_hour_rows": int((~log.hour_of_day.between(0, 23)).sum()),
        "invalid_date_rows": int(parsed_dates.isna().sum()), "nonfinite_numeric_values": numeric_nonfinite,
        "nonpositive_duration_tracks": int((tracks.duration <= 0).sum()),
    }
    observations = {
        "history_rows": len(history), "prediction_window_rows": len(log)-len(history),
        "sessions_without_non_skip2_history": int(history.groupby("session_id").skip_2.all().sum()),
        "sessions_without_complete_play_history": int((~history.groupby("session_id").not_skipped.any()).sum()),
        "non_skip2_not_complete_play_rows": int((~log.skip_2 & ~log.not_skipped).sum()),
        "single_play_tracks": int((plays.plays == 1).sum()), "tracks_with_at_least_five_plays": int((plays.plays >= 5).sum()),
        "sessions_with_changing_context": int((log.groupby("session_id").context_type.nunique() > 1).sum()),
        "sessions_with_changing_premium": int((log.groupby("session_id").premium.nunique() > 1).sum()),
        "zero_tempo_tracks": int((tracks.tempo == 0).sum()), "zero_time_signature_tracks": int((tracks.time_signature == 0).sum()),
        "flatness_above_one_tracks": int((tracks.flatness > 1).sum()), "positive_loudness_tracks": int((tracks.loudness > 0).sum()),
        "zero_norm_acoustic_vectors": int((np.linalg.norm(tracks[[f"acoustic_vector_{i}" for i in range(8)]], axis=1) < 1e-12).sum()),
        "popularity_min": float(tracks.us_popularity_estimate.min()), "popularity_max": float(tracks.us_popularity_estimate.max()),
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold",
        "axes.edgecolor": "#c5cfc1", "axes.grid": True, "grid.color": "#e4e9df", "grid.alpha": .7,
        "axes.axisbelow": True, "figure.facecolor": "#f7f8f4", "axes.facecolor": "#f7f8f4"})
    charts = []

    def save(fig, key, title, caption, insight, data, alt):
        filename = key + ".png"
        fig.savefig(image_dir / filename, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts.append({"id": key, "title": title, "caption": caption, "insight": insight, "alt": alt,
            "image_path": f"python/01_data_understanding/results/images/{filename}",
            "image_url": f"../../python/01_data_understanding/results/images/{filename}", "data": data})

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    bars = ax.bar(length_counts.index, length_counts.values, color=[ORANGE if x == 20 else GREEN for x in length_counts.index])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(xlabel="Tracks in a session", ylabel="Sessions", title="Recorded session lengths", xticks=length_counts.index, ylim=(0, length_counts.max()*1.18))
    save(fig, "session_lengths", "Session lengths", "Each session is counted once. The maximum recorded session length in these files is 20.",
        f"{length_counts.get(20, 0):,} of {len(sessions):,} sessions ({length_counts.get(20, 0)/len(sessions):.1%}) are at the 20-track maximum.",
        [{"length": int(k), "sessions": int(v)} for k,v in length_counts.items()],  "Bar chart of session counts for lengths 10 through 20, with the 20-track maximum highlighted.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, frame, key, label in [(axes[0],rate_length,"session_length","Session length"),(axes[1],rate_position,"session_position","Position within session")]:
        ax.plot(frame[key], frame.rate, marker="o", color=GREEN)
        ax.set(xlabel=label,ylabel="skip_2 rate",ylim=(0,1),xticks=frame[key][::2])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title("By length" if key == "session_length" else "By position")
    save(fig,"skip_rates","Skip rate by length and position", "Track-play-weighted skip_2 rates. Denominators vary by position; later positions only include longer sessions.",
        "Differences are descriptive associations, not evidence that session length or position causes skipping.",
        {"by_length":records(rate_length),"by_position":records(rate_position)}, "Two line charts of skip_2 rate by recorded session length and track position, with fixed 0–100% axes.")

    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    labels = [f"{r.hist_user_behavior_reason_start}  (n={r.observations:,})" for r in reason_rates.itertuples()]
    ax.barh(labels,reason_rates.rate,color=GREEN)
    ax.set(xlabel="skip_2 rate",xlim=(0,1),title="Skip rate by reason playback started")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    save(fig,"playback_start","Playback-start reasons", "Each bar uses track plays with that start reason. Small categories can have unstable rates. Future start reasons are not permitted model features.",
        "Playback-start reason is useful for descriptive understanding; using a future row's reason would break the observation boundary.",
        records(reason_rates), "Horizontal bars show skip_2 rates by start reason, with each category's play count in its label.")

    fig, axes = plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    contexts = context_rates.context_type.str.replace("_"," ")
    axes[0].barh(contexts,context_rates.observations,color=GREEN)
    axes[0].set(xlabel="Track plays",title="Where listening happens")
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1].barh(contexts,context_rates.rate,color=ORANGE)
    axes[1].set(xlabel="skip_2 rate",xlim=(0,1),title="Skip rate within each context")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    save(fig,"listening_contexts","Listening contexts", "Counts and rates are over track plays, not unique users or sessions. One session can contain multiple contexts.",
        f"{observations['sessions_with_changing_context']:,} sessions contain more than one context type; future context cannot be assumed constant.",
        records(context_rates), "Two horizontal bar charts compare play counts and skip_2 rates across listening contexts.")

    fig, axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    axes[0].plot(switch_position.session_position,switch_position.rate,marker="o",color=GREEN)
    switch_limit = max(.05, float(max(switch_position.rate.max(), switch_source.rate.max())) * 1.2)
    axes[0].set(xlabel="Position within session",ylabel="Context-switch rate",title="Context changes by position",ylim=(0,switch_limit),xticks=range(1,21,2))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].barh(switch_source.source_context.str.replace("_"," "),switch_source.rate,color=GREEN)
    axes[1].set(xlabel="Context-switch rate",xlim=(0,switch_limit),title="Context changes by previous context")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    save(fig,"context_switches","Context switches", "The source-context chart uses the previous track's context and excludes first positions. The position chart includes recorded first-position flags whose pre-session context is unavailable.",
        "Current context is not necessarily the source of a switch. Source categories are explicitly shifted within each session.",
        {"by_position":records(switch_position),"by_source_context":records(switch_source),"by_length":records(switch_length)}, "Line chart of context-switch rates by position and bar chart by previous context.")

    fig, axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    histogram_data={}
    for ax,subset,name in [(axes[0],plays,"All tracks"),(axes[1],plays.loc[plays.plays>=5],"At least five plays")]:
        values=subset.skip_rate.dropna().astype(float)
        counts,edges=np.histogram(values,bins=np.linspace(0,1,21))
        ax.hist(values,bins=edges,color=GREEN,edgecolor="#f7f8f4")
        ax.set(xlabel="Per-track skip_2 rate",ylabel="Unique tracks",title=f"{name} (n={len(values):,})",xlim=(0,1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        histogram_data[name]={"tracks":len(values),"bin_edges":edges.tolist(),"counts":counts.tolist()}
    save(fig,"track_skip_distribution","Per-track skip rates", "Each track contributes one rate, independent of how often it was played. The filtered view requires at least five observed plays; it changes the population.",
        f"{observations['single_play_tracks']:,} tracks ({observations['single_play_tracks']/len(plays):.1%}) appear once. Their observed rates can only be 0% or 100%.",
        histogram_data, "Histograms compare all tracks' observed skip rates with the subset having at least five plays.")

    fig,axes=plt.subplots(2,4,figsize=(14,7),layout="constrained")
    feature_histograms={}
    for ax,col in zip(axes.flat,FEATURES):
        values=tracks[col].replace([np.inf,-np.inf],np.nan).dropna()
        counts,edges=np.histogram(values,bins=30)
        ax.hist(values,bins=edges,color=GREEN,edgecolor="#f7f8f4",linewidth=.4)
        ax.set(title=col.replace("_"," ").title(),xlabel=col,ylabel="Unique tracks")
        ax.tick_params(labelsize=8)
        feature_histograms[col]={"n":len(values),"bin_edges":edges.tolist(),"counts":counts.tolist()}
    save(fig,"acoustic_features","Acoustic-feature distributions", "Each metadata track is counted once. These are feature distributions, not feature importance or causal effects.",
        "Features have different scales and shapes. These differences matter for the preprocessing: the linear models scale the features, and the boosted trees do not need scaling.",
        feature_histograms, "Eight histograms show acousticness, danceability, energy, instrumentalness, speechiness, valence, tempo, and loudness over unique tracks.")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    counts,edges=np.histogram(tracks.us_popularity_estimate.dropna(),bins=30)
    axes[0].hist(tracks.us_popularity_estimate,bins=edges,color=GREEN)
    axes[0].set(xlabel="US popularity estimate",ylabel="Unique tracks",title="Popularity in this selected sample")
    bins=[0,1,2,4,9,49,np.inf]; names=["1","2","3–4","5–9","10–49","50+"]
    coverage=pd.cut(plays.plays,bins=bins,labels=names).value_counts(sort=False)
    axes[1].bar(names,coverage.values,color=ORANGE)
    axes[1].set(xlabel="Observed plays per track",ylabel="Unique tracks",title="How often is each track observed?")
    save(fig,"popularity_coverage","Popularity and track coverage", "Both panels count unique tracks. Popularity is the supplied estimate; play counts refer only to this mini log.",
        f"Popularity ranges from {observations['popularity_min']:.2f} to {observations['popularity_max']:.2f}; the sample does not span an unrestricted music catalog.",
        {"popularity":{"bin_edges":edges.tolist(),"counts":counts.tolist()},"play_count_buckets":[{"bucket":str(k),"tracks":int(v)} for k,v in coverage.items()]}, "Histogram of the supplied popularity estimate and bars of track coverage by observed play-count bucket.")

    issues=[
        {"id":"empty_history","title":"Some history profiles are empty","count":observations['sessions_without_non_skip2_history'],"unit":"sessions","severity":"handling_required","detail":"No first-half track has skip_2=false. Distances to the kept-song profile are left missing for these sessions, and a missing-profile indicator is added."},
        {"id":"different_targets","title":"Non-skip_2 is not complete playback","count":observations['non_skip2_not_complete_play_rows'],"unit":"plays","severity":"definition","detail":"These rows have both skip_2=false and not_skipped=false. The two terms are kept separate in this project."},
        {"id":"sparse_tracks","title":"Many tracks have one observed play","count":observations['single_play_tracks'],"unit":"tracks","severity":"interpretation","detail":"Track-level skip rates based on very few plays are unstable; we show the number of plays behind each rate and label every filtered view."},
        {"id":"changing_context","title":"Context can change inside a session","count":observations['sessions_with_changing_context'],"unit":"sessions","severity":"leakage_risk","detail":"Use only observed first-half context when predicting the future."},
        {"id":"changing_premium","title":"Premium status is not always constant","count":observations['sessions_with_changing_premium'],"unit":"sessions","severity":"handling_required","detail":"These rows are kept, and the last first-half value is used, never a later one."},
        {"id":"zero_tempo","title":"Zero tempo values occur","count":observations['zero_tempo_tracks'],"unit":"tracks","severity":"investigate","detail":"The original values are kept. Zero may mean the descriptor was not estimated, but these files do not confirm this."},
        {"id":"zero_meter","title":"Zero time signatures occur","count":observations['zero_time_signature_tracks'],"unit":"tracks","severity":"investigate","detail":"Zero is not read as a real musical meter and is not replaced by any made-up value."},
        {"id":"feature_ranges","title":"Not every acoustic feature is a probability","count":observations['flatness_above_one_tracks'],"unit":"tracks with flatness > 1","severity":"interpretation","detail":"Signed vectors, positive loudness and flatness above one are left as they are and are not clipped to [0,1]."},
    ]
    structural_problem_count = sum(checks[k] for k in ["duplicate_log_rows","duplicate_track_rows","duplicate_session_positions","duplicate_track_ids","invalid_sessions","unmatched_log_tracks_rows","invalid_hour_rows","invalid_date_rows","nonpositive_duration_tracks"])
    structural_problem_count += sum(sum(v.values()) for v in checks['missing_by_column'].values()) + sum(checks['invalid_indicator_counts'].values()) + sum(checks['negative_seek_counts'].values()) + sum(sum(v.values()) for v in numeric_nonfinite.values())
    payload={
        "schema_version":"1.0", "generated_at":datetime.now(timezone.utc).isoformat(),
        "title":"Understanding the Music Streaming Sessions mini dataset", "status":"computed_from_local_data",
        "scope":{"population":"All rows in the two supplied mini CSV files", "purpose":"Descriptive data understanding only; no models or test metrics", "target":"skip_2", "split_status":"No train/test split is created or used by this script", "caveat":"We explored the full mini dataset before fixing the train/test split, so the test sessions were not completely unseen during this descriptive step. No model choice was based on these plots; every search and choice used development data only."},
        "paths":{"project_root":"ml-project", "json":"python/01_data_understanding/results/data_understanding.json", "image_path_base":"Project root", "image_url_base":"website/html/data.html", "fetch_from_data_page":"../../python/01_data_understanding/results/data_understanding.json"},
        "sources":[
            {"title":"Spotify Sequential Skip Prediction Challenge","url":"https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge","role":"Official challenge and access information"},
            {"title":"Mini-data archive named in the project","url":"https://github.com/suhani6/spotify-skip-analysis","role":"Public archive that the two mini files were taken from"}],
        "access_limits":["The official challenge page carries a July 2024 notice that original downloads are no longer available and directs requests to Spotify Research.","This analysis uses the two CSV files in the data/ folder, taken from a public archive. The original dataset's terms are those of the WSDM Cup 2019 release; the files are included only so that the results can be reproduced.","The local sample contains sessions of 10–20 tracks and a narrow popularity range. Findings should not be generalized to all streaming sessions or the full music catalog.","The supplied mini files contain no song titles, cross-session user IDs, uniform_random indicator or playlist snapshots. Shuffle alone cannot establish random treatment assignment."],
        "size":{"track_plays":len(log),"sessions":len(sessions),"unique_logged_tracks":int(log.track_id_clean.nunique()),"metadata_tracks":len(tracks),"log_columns":len(log.columns),"track_columns":len(tracks.columns),"session_length_min":int(sessions.recorded_length.min()),"session_length_max":int(sessions.recorded_length.max()),"distinct_dates":int(parsed_dates.nunique()),"date_min":parsed_dates.min().date().isoformat(),"date_max":parsed_dates.max().date().isoformat(),"note":"Date endpoints do not imply coverage of every intervening day."},
        "tables":[{"file":"log_mini.csv","row_unit":"One track play within a session","rows":len(log),"columns":len(log.columns),"size_bytes":(args.data_dir/'log_mini.csv').stat().st_size}, {"file":"tf_mini.csv","row_unit":"One unique track","rows":len(tracks),"columns":len(tracks.columns),"size_bytes":(args.data_dir/'tf_mini.csv').stat().st_size}],
        "join":{"left":"log.track_id_clean","right":"track_features.track_id","cardinality":"many-to-one","unmatched_rows":checks['unmatched_log_tracks_rows']},
        "important_variables":[dict(zip(["name","table","meaning","relevance"],x)) for x in IMPORTANT],
        "columns":{"log":schema_summary(log),"track_features":schema_summary(tracks)},
        "target_summary":{c:{"positive_count":int(log[c].sum()),"labelled_plays":int(log[c].count()),"positive_rate":float(log[c].mean())} for c in ["skip_1","skip_2","skip_3","not_skipped"]},
        "data_quality":{"status":"no_structural_problems_detected" if structural_problem_count==0 else "review_required", "checks":checks,"observations":observations,"issues":issues,"action":"No cleaning, imputation, feature fitting or raw-file edits performed."},
        "charts":charts,
        "key_findings":[{"title":"Recorded sessions concentrate at the cap","text":charts[0]['insight']}, {"title":"Track rates need denominators","text":charts[5]['insight']}, {"title":"Popularity covers a narrow range","text":charts[7]['insight']}, {"title":"Empty profiles need explicit handling","text":f"{observations['sessions_without_non_skip2_history']:,} sessions have no non-skip_2 track in the first half."}],
        "reproduce":{"command":"python3 python/01_data_understanding/analyze.py","requirements":"python/01_data_understanding/requirements.txt","python_minimum":"3.10","libraries":{"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__}},
    }
    # Reject nonstandard NaN/Infinity output: browsers can parse this as ordinary JSON.
    json_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {len(charts)} plots to {image_dir}")
    print(f"Saved summaries and image paths to {json_path}")
    print(f"Data quality: {payload['data_quality']['status']}")


if __name__ == "__main__":
    main()
