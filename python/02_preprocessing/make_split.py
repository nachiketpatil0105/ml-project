"""Step 2a. Create the fixed 80/20 split of sessions (development / test) and save it.

Run from the project root:  python3 python/02_preprocessing/make_split.py
Output: results/session_split.json  (the exact session ids of each side, seed 42)
If the file already exists it is checked against the data and left unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from features import load_data, validate, session_split

log, tracks = load_data()
validate(log, tracks)
split = session_split(log.session_id)
print("development sessions:", len(split["development"]), "test sessions:", len(split["test"]), "settings:", split["settings"])
