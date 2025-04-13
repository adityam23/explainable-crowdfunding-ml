# Explainable-crowdfunding-ml
Crowdfunding dataset analysis using xai techniques

This repo is trying to replicate the results for [this paper](https://www.mdpi.com/2071-1050/17/4/1361) using only the [Web Robots dataset](https://webrobots.io/kickstarter-datasets), which is much smaller in terms of text.

# Installation

1. Create virtual environment
    ```
    python -m venv .venv
    ```

2. Use virtual environment and install requirements
    ```python
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

This repo has been tested using Python 3.13 on linux.

# Running the project

The dataset is composed of csv files which are compressed in zip format. Each month is present as one zip file, with multiple csv files contained within it. The subfolders are of the format `Kickstarter_<ISO_TIMESTAMP_WITH_UNDERSCORES>`. Each of these subfolders contains one or more CSV files named:
- Kickstarter.csv
- Kickstarter2.csv
- ...
- Kickstarter64.csv

The script `prepare_dataset.py` creates a single file for use with the notebook `Explanations.ipynb`. It expects a folder with the following structure:
```
data/
├── Kickstarter_2024-04-15T06_47_07_694Z/
│   ├── Kickstarter.csv
│   ├── Kickstarter2.csv
│   └── Kickstarter64.csv
├── Kickstarter_2024-03-10T13_22_45_123Z/
│   ├── Kickstarter.csv
│   ├── Kickstarter7.csv
│   └── Kickstarter42.csv
└── Kickstarter_2024-02-01T09_05_12_999Z/
    ├── Kickstarter.csv
    └── Kickstarter10.csv
```
You can optionally provide path to data folder and the final output filename.

`Explanations.ipynb` is the jupyter notebook containing the training and comparison of the ML model, as well as SHAP and Lime explanations for the dataset.

The explanations presented in the notebook are for projects between 2024-01 and 2025-03.

The file `full_dataset.csv.xz` contains an xz compressed version of the complete dataset (single file you can use for Explanations). To extract the file, xz-utils package is needed.

Run the command
```
unxz full_dataset.csv.xz
```
in order to uncompress the file on Unix-like devices, or use 7zip on Windows devices.

# Issue while running Lime experiments

In case you encounter issue while running Lime with newer versions of python, you can go to `.venv/lib/python3.13/site-packages/lime/explanation.py` in line 194, change

```python
from IPython.core.display import display, HTML
```

to

```python
from IPython.display import display, HTML
```
