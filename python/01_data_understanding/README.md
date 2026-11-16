# Data understanding

From the `ml-project` root:

```bash
python3 -m pip install -r python/01_data_understanding/requirements.txt
python3 python/01_data_understanding/analyze.py
```

The script reads `data/log_mini.csv` and `data/tf_mini.csv`. It does not change the CSVs, train models, create splits, or verify hashes. Dataset source links are included in the output.

Outputs (in `results/` next to this script):

- `results/data_understanding.json`: sources, access limits, size, variable descriptions, per-column summaries, target frequencies, quality checks, chart data, findings and image paths.
- `results/images/`: eight PNG figures.

The Data page of the website (`website/html/data.html`) reads this JSON and the images directly.
