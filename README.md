# Week 1: Data Acquisition, Cleaning, and Preprocessing

This repository contains Python source code for the Week 1 data-cleaning task using the Titanic passenger dataset.

## Dataset
- File: `Titanic-Dataset.csv`
- Source: Titanic passenger data commonly used for data-analysis practice. If your assignment requires a source URL, add the exact page from which you obtained this CSV and verify its license/terms before publishing.

## What the code does
1. Loads the CSV and checks its dimensions.
2. Reports data types, summary statistics, missing values, and duplicate rows.
3. Trims whitespace and standardizes empty strings.
4. Converts expected numeric fields and checks invalid values in `Age`, `Fare`, `Survived`, `Pclass`, `Sex`, and `Embarked`.
5. Removes exact duplicate rows.
6. Imputes numeric missing values using medians and categorical missing values using modes.
7. Flags potential outliers using the IQR rule without deleting them automatically.
8. Creates a model-ready dataset using one-hot encoding and standardization.
9. Saves cleaned data, preprocessed data, and a text summary in `output/`.

## Setup
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
Keep `Titanic-Dataset.csv` in the same folder as `data_cleaning.py`, then run:

```bash
python data_cleaning.py
```

## Output files
- `output/cleaned_titanic.csv` — cleaned dataset with original columns retained.
- `output/preprocessed_titanic.csv` — encoded and scaled dataset for later analysis.
- `output/cleaning_summary.txt` — initial exploration and a record of cleaning choices.

## Important notes
- Outliers are flagged, not automatically removed, because unusual values can be valid observations.
- Imputation choices can affect distributions and downstream results; document these trade-offs in the report.
- For machine-learning evaluation, split into train/test sets first and fit imputers/scalers on the training set only to avoid data leakage. This assignment script standardizes the full dataset for demonstration, not for final model evaluation.
- Do not upload private data or credentials to GitHub. Check the dataset's source and licensing before making the CSV public.
