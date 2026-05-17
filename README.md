# Explainable Crowdfunding ML

Crowdfunding dataset analysis using XAI techniques (SHAP, LIME, XGBoost).

This project replicates the results from [this paper](https://www.mdpi.com/2071-1050/17/4/1361) using the [Web Robots Kickstarter dataset](https://webrobots.io/kickstarter-datasets).

## Setup

**Requirements:** Python 3.14+, [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adityam23/explainable-crowdfunding-ml.git
cd explainable-crowdfunding-ml
uv sync
```

`uv sync` installs both runtime and dev dependencies (including `ruff`). To skip dev tooling, use `uv sync --no-dev`.

### Lint and format

```bash
uv run ruff format .     # auto-format
uv run ruff check .      # lint
uv run ruff check --fix . # lint + autofix
```

Configuration lives in `pyproject.toml` under `[tool.ruff]`. Notebooks are excluded from ruff by default.

## Running the project

The dataset is composed of CSV files compressed in zip format, organized by month in subfolders like `Kickstarter_<ISO_TIMESTAMP>`. Each subfolder contains kickstarter CSV files.

`prepare_dataset.py` creates a single file for use with the notebooks. It expects:

```
data/
  Kickstarter_2024-04-15T06_47_07_694Z/
    Kickstarter.csv
    Kickstarter2.csv
    Kickstarter64.csv
  Kickstarter_2024-03-10T13_22_45_123Z/
    Kickstarter.csv
    Kickstarter7.csv
    Kickstarter42.csv
  Kickstarter_2024-02-01T09_05_12_999Z/
    Kickstarter.csv
    Kickstarter10.csv
```

You can optionally provide a path to the data folder and the final output filename.

`Explanations.ipynb` contains the training and comparison of the ML model, plus SHAP and Lime explanations for the dataset. `Explanations-Dice.ipynb` adds DICE counterfactual explanations.

The explanations cover projects between 2024-01 and 2025-03.

### Compressed dataset

The file `full_dataset.csv.xz` contains an xz compressed version of the complete dataset. To extract:

```bash
unxz full_dataset.csv.xz
```

Alternatively, download from Google Drive: https://drive.google.com/file/d/1pt_b1G5oXA6ERSmdlCc_EXhiOCVMIGwX/view?usp=sharing

## Known issues

If Lime encounters an import error with newer Python versions, edit `.venv/lib/python3.13/site-packages/lime/explanation.py` line 194, changing:

```python
from IPython.core.display import display, HTML
```

to:

```python
from IPython.display import display, HTML
```
