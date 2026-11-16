# Predicting skips in the second half of a music streaming session (ES 335)

**Question.** Can classical machine learning predict which songs a listener will skip in the second half of a streaming session, using only the first half of the session and numeric features of the upcoming songs?
One row = one song in the second half of a session. Target = `skip_2`. Main score = Mean Average Accuracy (MAA), as in the WSDM Cup 2019 challenge.
The model is **feature-based**: it uses first-half behaviour and numeric song features, not song titles, artists or song identity.

**Live site:** https://nachiketpatil0105.github.io/ml-project · **Repository:** https://github.com/nachiketpatil0105/ml-project · Authors: Patil Nachiket Kiran (24110250), Krish Kishor Patil (24110171).

**Website:** `index.html` and `website/`. The pages are plain HTML, CSS and JavaScript. They read all numbers and tables from the result files in `python/*/results/`, so the website and the code always agree. Serve the folder over HTTP (`python3 -m http.server`) or publish it with GitHub Pages; opening the files directly from disk does not work because browsers block file loading.

## Data access
`data/log_mini.csv` (session log, about 168,000 rows) and `data/tf_mini.csv` (song features, about 50,000 songs) are the mini partition of the Music Streaming Sessions Dataset (Brost et al., WWW 2019; WSDM Cup 2019). The official download is no longer offered, so they come from the public archive `https://github.com/suhani6/spotify-skip-analysis`. No script changes them.

## Setup and run
Python 3.10 or newer. From this folder:

    python3 -m pip install -r requirements.txt
    python3 python/01_data_understanding/analyze.py
    python3 python/02_preprocessing/make_split.py
    python3 python/02_preprocessing/describe_features.py
    python3 -m unittest discover -s python/02_preprocessing -v
    python3 python/03_tuning/run_experiments.py --trials 40 --top 5
    python3 python/03_tuning/make_plots.py
    python3 python/04_final_test/run_final.py
    python3 python/04_final_test/make_plots.py
    python3 python/05_error_analysis/analyze_errors.py
    python3 python/05_error_analysis/make_plots.py
    python3 python/06_explorer/export_explorer.py
    python3 -m http.server

Step 3 is the slow one (30 to 90 minutes). It saves its progress (`stages/`, `optuna.db`), so an interrupted run continues; add `--force` to start over. `SKIP_JOBS=<n>` sets the CPU cores the models use. `QUICK=1` runs any step on a small subset into `results_quick/` folders as a smoke test.
We used short Python scripts instead of notebooks. Each script prints what it does and writes its results next to itself.

## Layout: each step keeps its own results
| Folder | What it does | Results it writes (in its `results/` folder) |
|---|---|---|
| `python/01_data_understanding` | Describes the raw data | `data_understanding.json`, `images/` |
| `python/02_preprocessing` | Makes the fixed 80/20 session split, describes the features, tests for leakage | `session_split.json`, `feature_summary.json` (allowed input columns), `features_preview.csv` |
| `python/03_tuning` | Optuna search and repeated grouped CV on development sessions only; ablation; threshold | `development.json`, `frozen_config.json`, `cv_folds.json`, `development_oof_primary.csv`, `optuna.db`, `stages/`, plots |
| `python/04_final_test` | Refits on development data and scores the test sessions once | `test_results.json`, `test_predictions.csv`, `lightgbm_model.txt`, plot |
| `python/05_error_analysis` | Error analysis of the main model | `error_analysis.json`, plots |
| `python/06_explorer` | Exports the model and sessions for the in-browser explorer | `model.json`, `sessions.json` |
| `python/shared` | Shared code: features, models, metrics, cross-validation, paths | – |
| `website/` | Static pages, styles and scripts (`html/`, `css/`, `js/`) | – |

## Method in brief
- A session has L songs. The first ceil(L/2) are the history (odd lengths round up, as in the challenge). The rest are predicted. Features use only the history, the song's numeric features and its queue position. A unit test checks that second-half answers never reach the features.
- Sessions are split 80/20 by session id (seed 42): 8,000 development and 2,000 test sessions.
- Baselines: training-label majority, first-half skip majority, copy last skip, popularity-only logistic regression. Models: logistic regression, random forest, LightGBM, XGBoost, CatBoost.
- Tuning: Optuna (TPE), 40 trials per boosting library, stratified session-grouped 5-fold CV; the best 5 trials are re-scored on three more repeats of 5-fold CV and the choice is made on that. The test set is scored once by step 4.
- Scores: MAA (main), first-prediction accuracy, ROC AUC; 95% session-bootstrap intervals and paired comparisons.

## Results of the submitted run
Test sessions (2,000): LightGBM MAA 0.571 (95% interval 0.557 to 0.583) against 0.537 for copy last skip; difference +0.034 (0.023 to 0.045). Logistic regression 0.552, random forest 0.559, XGBoost 0.570, CatBoost 0.572. The three boosting libraries cannot be told apart. Listening history carries most of the signal; popularity and audio similarity add little. Full tables are on the website and in `python/04_final_test/results/test_results.json` and `python/03_tuning/results/development.json`.
Please note: LightGBM and the other libraries can give slightly different models on different computers (core count, library version), so a re-run may change scores in the third decimal place.

## Extra experiments
All Optuna trials and tuning grids are in `python/03_tuning/results/` (`development.json`, `optuna.db`). The feature-group ablation and the threshold curve are in `development.json`. Earlier development runs (first half = floor(L/2), development folds only): a per-song skip-rate feature was dropped (MAA 0.574 against 0.577); `skip_1`/`skip_3` features and per-session audio deltas changed MAA by under 0.001 and were dropped; removing the start/end reason shares lowered MAA from 0.577 to 0.560. These exploratory runs were done by hand with an earlier pipeline version and are described, not scripted: their numbers cannot be re-created from this repository. Everything else (Optuna search, grids, ablation, threshold curve) is scripted and saved.

## Use of the test set (disclosure)
During development the test sessions were scored once with an earlier version of the pipeline (first half = floor(L/2), small grids): LightGBM 0.562, copy last skip 0.531. We then changed the first-half rule to match the challenge and replaced the grids with the Optuna search. Every search and choice used development data only. The final pipeline scored the test sessions once; the earlier run is disclosed here and on the Evaluation page.

## AI-use disclosure
We used Claude (Anthropic) and ChatGPT (OpenAI) as a coding and writing assistant. It suggested the feature set, the comparison of LightGBM, XGBoost and CatBoost, the threshold rule and the explorer. We set the research question and proposal, decided the first-half rule, the search method and the validation method, reviewed and approved each change, and are responsible for the submission. All numbers on the website come from the scripts in this repository.

## References
- Brost, Mehrotra, Jehan. The Music Streaming Sessions Dataset. WWW 2019. WSDM Cup 2019 Spotify Sequential Skip Prediction Challenge (AIcrowd).
- Ferraro et al. Skip prediction using boosting trees based on acoustic features of tracks in sessions. arXiv:1903.11833.
- Béres, Kelen, Benczúr. Sequential skip prediction using deep learning and ensembles. WSDM Cup 2019. https://core.ac.uk/download/228079504.pdf
- Modelling Sequential Music Track Skips using a Multi-RNN Approach. arXiv:1903.08408.
- Adapa. Sequential modeling of sessions using recurrent neural networks for skip prediction. arXiv:1904.10273.
- Akiba et al., Optuna, KDD 2019; Ke et al., LightGBM, NeurIPS 2017; Chen and Guestrin, XGBoost, KDD 2016; Prokhorenkova et al., CatBoost, NeurIPS 2018; Pedregosa et al., scikit-learn, JMLR 2011.
