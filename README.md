# Predicting skips in the second half of a music streaming session (ES 335)

**Question.** Can classical machine learning predict which songs a listener will skip in the second half of a streaming session, using only the first half of the session and numeric features of the upcoming songs?
One row = one song in the second half of a session. Target = `skip_2`. Main score = Mean Average Accuracy (MAA), as in the WSDM Cup 2019 challenge.
The model is **feature-based**: it uses first-half behaviour and numeric song features, not song titles, artists or song identity.

**Live site:** https://nachiketpatil0105.github.io/ml-project · **Repository:** https://github.com/nachiketpatil0105/ml-project · Authors: Patil Nachiket Kiran (24110250), Krish Kishor Patil (24110171).

**Website:** `index.html` and `website/`. The pages are plain HTML, CSS and JavaScript. They read all numbers and tables from the result files in `python/*/results/`, so the website and the code always agree. Serve the folder over HTTP (`python3 -m http.server`) or publish it with GitHub Pages; opening the files directly from disk does not work because browsers block file loading.

## Data access
`data/log_mini.csv` (session log, about 168,000 rows) and `data/tf_mini.csv` (song features, about 50,000 songs) are the mini partition of the Music Streaming Sessions Dataset (Brost et al., WWW 2019; WSDM Cup 2019). The official download is no longer offered, so they come from the public archive `https://github.com/suhani6/spotify-skip-analysis`. No script changes them.

## Setup and run
Python 3.12 or newer (the submitted run used Python 3.13 and the exact versions in `requirements.txt`). From this folder:

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
    python3 python/07_extra_experiments/run_extra.py
    python3 -m http.server

Step 3 is the slow one (about 1 hour 45 minutes on 2 CPU cores for the submitted run; less with more cores). It saves its progress (`stages/`, `optuna.db`), so an interrupted run continues; add `--force` to start over. `SKIP_JOBS=<n>` sets the CPU cores the models use. `QUICK=1` runs any step on a small subset into `results_quick/` folders as a smoke test.
The scripts do the work; each prints what it does and writes its results next to itself. `notebooks/walkthrough.ipynb` loads the saved results and walks through the steps in order without re-training anything (needs `jupyter`, not in `requirements.txt`).

## Layout: each step keeps its own results
| Folder | What it does | Results it writes (in its `results/` folder) |
|---|---|---|
| `python/01_data_understanding` | Describes the raw data | `data_understanding.json`, `images/` |
| `python/02_preprocessing` | Makes the fixed 80/20 session split, describes the features, tests for leakage | `session_split.json`, `feature_summary.json` (allowed input columns), `features_preview.csv`, `unittest_output.txt` (output of the leakage and metric tests) |
| `python/03_tuning` | Optuna search and repeated grouped CV on development sessions only; ablation; threshold | `development.json`, `frozen_config.json`, `cv_folds.json`, `development_oof_primary.csv`, `optuna.db`, `stages/`, plots |
| `python/04_final_test` | Refits on development data and scores the test sessions once | `test_results.json`, `test_predictions.csv`, `lightgbm_model.txt`, plot |
| `python/05_error_analysis` | Error analysis of the main model | `error_analysis.json`, plots |
| `python/06_explorer` | Exports the model and sessions for the in-browser explorer | `model.json`, `sessions.json` |
| `python/07_extra_experiments` | "If we change one thing": one-change variants of the final model on the same 15 development folds | `extra_experiments.json` |
| `notebooks/` | Walkthrough notebook that reads the saved results | – |
| `python/shared` | Shared code: features, models, metrics, cross-validation, paths | – |
| `website/` | Static pages, styles and scripts (`html/`, `css/`, `js/`) | – |

## Method in brief
- A session has L songs. The first ceil(L/2) are the history (odd lengths round up, as in the challenge). The rest are predicted. Features use only the history, the song's numeric features and its queue position. A unit test checks that second-half answers never reach the features.
- Sessions are split 80/20 by session id (seed 42): 8,000 development and 2,000 test sessions.
- Baselines: training-label majority, first-half skip majority, copy last skip, popularity-only logistic regression. Models: logistic regression, random forest, LightGBM, XGBoost, CatBoost.
- Tuning: Optuna (TPE), 40 trials per boosting library, stratified session-grouped 5-fold CV; the best 5 trials are re-scored on three more repeats of 5-fold CV and the choice is made on that. The test set is scored once by step 4.
- Scores: MAA (main), first-prediction accuracy, ROC AUC; 95% session-bootstrap intervals and paired comparisons.

## Results of the submitted run
Test sessions (2,000): LightGBM MAA 0.571 (95% interval 0.558 to 0.584) against 0.537 for copy last skip; difference +0.034 (0.023 to 0.045). Logistic regression 0.552, random forest 0.560, XGBoost 0.570, CatBoost 0.571. The three boosting libraries cannot be told apart. Listening history carries most of the signal; popularity and audio similarity add little. Full tables are on the website and in `python/04_final_test/results/test_results.json` and `python/03_tuning/results/development.json`.
Please note: LightGBM and the other libraries can give slightly different models on different computers (core count, library version), so a re-run may change scores in the third decimal place.

## Extra experiments
Everything is scripted and saved. The Optuna trials and tuning grids are in `python/03_tuning/results/` (`development.json`, `optuna.db`), together with the feature-group ablation and the threshold curve. `python/07_extra_experiments/run_extra.py` re-scores the final LightGBM with exactly one change at a time, on the same 15 development folds (test sessions never loaded): dropping the audio, similarity or popularity group; dropping the start/end reason shares; adding skip_1/skip_3 history, per-session audio deltas, or a per-song skip rate (rebuilt inside every training fold); swapping LightGBM for logistic regression; and using floor(L/2) instead of ceil(L/2). Results: `python/07_extra_experiments/results/extra_experiments.json`, shown on the Team and contributions page. Early hand-run versions of some of these, made with an older pipeline, are replaced by these scripted runs.

## Environment and random seeds
Python 3.13; numpy 2.2.6, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, optuna 5.1.0, matplotlib 3.10.7 (pinned in `requirements.txt`).

| What | Seed | Where |
|---|---|---|
| Development / test split of sessions | 42 | `shared/features.py` (`session_split`) |
| CV folds for the Optuna search | 42 | `shared/cv.py` (`SEARCH_SEED`) |
| CV folds for re-scoring, ablation, extra experiments | 101, 102, 103 | `shared/cv.py` (`REPEAT_SEEDS`) |
| Optuna TPE sampler | 42 | `03_tuning/run_experiments.py` |
| All models (`random_state` / `random_seed`) | 42 | `shared/models.py` (`SEED`) |
| Session bootstrap | 42 | `shared/metrics.py` |
| Sessions sampled for the explorer | 42 | `06_explorer/export_explorer.py` |
| Fake track features in unit tests | 0 | `02_preprocessing/test_features.py` |

## Data, privacy and credentials
- No personal information: the data has no names, user IDs, song titles or artists. Session and track IDs are anonymous codes that come with the dataset and cannot be linked to a person or across sessions.
- No credentials: no API, keys, passwords or tokens are used or stored, and nothing is downloaded at run time.
- The two CSV files are included only to make the results reproducible, because the official download is no longer offered. They stay under the terms of the original WSDM Cup 2019 release and can also be taken from the public archive above.
- The website is static; the explorer runs in the visitor's browser and sends nothing anywhere.

## Licence
Code, website and text: MIT (`LICENSE`). The licence does not cover `data/`, which keeps the terms of its original release.

## Contributions
We split the coding and writing about equally; the full table, with each person's key choices, checks, a limitation and a "change one thing" example, is on the Team and contributions page (`website/html/team.html`).
- **Patil Nachiket Kiran (24110250):** data checks and exploration (step 1), session split, feature code, first-half rule implementation and leakage tests (step 2, `shared/features.py`), error analysis (step 5), website frame and loader (`site.js`, CSS), walkthrough notebook.
- **Krish Kishor Patil (24110171):** model, metric and CV code (`shared/models.py`, `metrics.py`, `cv.py`), Optuna search, repeated CV, ablation and threshold (step 3), final test and bootstrap (step 4), in-browser explorer (step 6), "change one thing" experiments (step 7).
- **Together:** research question, proposal and dataset choice; evaluation plan (MAA, split by session, grouped CV) and the first-half rule decision; reading the results; Limitations, Conclusion and README; final review of every page.

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
