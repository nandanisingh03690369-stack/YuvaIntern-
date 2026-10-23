"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: Titanic passenger dataset (Titanic-Dataset.csv)

Run:
    pip install -r requirements.txt
    python data_cleaning.py

Outputs are written to the output/ folder:
    - cleaned_titanic.csv
    - preprocessed_titanic.csv
    - cleaning_summary.txt

Note:
This script keeps the original dataset unchanged and writes processed copies.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Titanic-Dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV dataset and validate that it contains rows and columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place Titanic-Dataset.csv in the same folder as data_cleaning.py."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset loaded successfully but contains no rows.")
    return df


def explore_data(df: pd.DataFrame) -> str:
    """Create a concise initial data-quality report."""
    lines = [
        "WEEK 1: DATA ACQUISITION, CLEANING, AND PREPROCESSING",
        "=" * 58,
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Column data types:",
        df.dtypes.to_string(),
        "",
        "Missing values by column:",
        df.isna().sum().to_string(),
        "",
        "Numeric summary:",
        df.describe(include=[np.number]).to_string(),
        "",
        "Categorical summary:",
        df.describe(include=["object"]).to_string(),
    ]
    return "\n".join(lines)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean missing values, duplicate rows, invalid categories, and outliers."""
    df = df.copy()
    notes = []

    # Normalize whitespace in text columns and standardize empty strings as missing.
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[col] = df[col].replace("", np.nan)
    notes.append("Trimmed whitespace and converted empty text values to missing values.")

    # Remove exact duplicate records.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    notes.append(f"Removed {duplicate_count} exact duplicate row(s).")

    # Convert expected numeric columns safely; unexpected text becomes missing.
    numeric_candidates = [
        col for col in ["PassengerId", "Survived", "Pclass", "Age",
                        "SibSp", "Parch", "Fare"]
        if col in df.columns
    ]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    notes.append("Converted expected numeric columns to numeric data types.")

    # Correct impossible values using domain knowledge.
    if "Age" in df.columns:
        invalid_age = int(((df["Age"] < 0) | (df["Age"] > 110)).sum())
        df.loc[(df["Age"] < 0) | (df["Age"] > 110), "Age"] = np.nan
        notes.append(f"Set {invalid_age} implausible Age value(s) to missing.")

    if "Fare" in df.columns:
        invalid_fare = int((df["Fare"] < 0).sum())
        df.loc[df["Fare"] < 0, "Fare"] = np.nan
        notes.append(f"Set {invalid_fare} negative Fare value(s) to missing.")

    if "Survived" in df.columns:
        invalid_survival = int((~df["Survived"].isin([0, 1])).sum())
        df.loc[~df["Survived"].isin([0, 1]), "Survived"] = np.nan
        notes.append(f"Set {invalid_survival} invalid Survived value(s) to missing.")

    if "Pclass" in df.columns:
        invalid_class = int((~df["Pclass"].isin([1, 2, 3])).sum())
        df.loc[~df["Pclass"].isin([1, 2, 3]), "Pclass"] = np.nan
        notes.append(f"Set {invalid_class} invalid Pclass value(s) to missing.")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower()
        invalid_sex = int((df["Sex"].notna() & ~df["Sex"].isin(["male", "female"])).sum())
        df.loc[df["Sex"].notna() & ~df["Sex"].isin(["male", "female"]), "Sex"] = np.nan
        notes.append(f"Set {invalid_sex} unexpected Sex category value(s) to missing.")

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].str.upper()
        invalid_port = int(
            (df["Embarked"].notna() & ~df["Embarked"].isin(["S", "C", "Q"])).sum()
        )
        df.loc[df["Embarked"].notna() & ~df["Embarked"].isin(["S", "C", "Q"]), "Embarked"] = np.nan
        notes.append(f"Set {invalid_port} unexpected Embarked value(s) to missing.")

    # Impute numeric missing values with medians and categorical values with modes.
    # Median is robust to skewed distributions; mode keeps categories valid.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        missing = int(df[col].isna().sum())
        if missing:
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)
                notes.append(f"Filled {missing} missing value(s) in {col} with median ({median_value:g}).")

    for col in categorical_cols:
        missing = int(df[col].isna().sum())
        if missing:
            mode_values = df[col].mode(dropna=True)
            if not mode_values.empty:
                mode_value = mode_values.iloc[0]
                df[col] = df[col].fillna(mode_value)
                notes.append(f"Filled {missing} missing value(s) in {col} with mode ({mode_value}).")
            else:
                df[col] = df[col].fillna("Unknown")
                notes.append(f"Filled {missing} missing value(s) in {col} with 'Unknown'.")

    # Flag outliers with the IQR method. Do not automatically delete them:
    # high fares can be genuine first-class tickets, so values are retained.
    outlier_summary = []
    for col in [c for c in ["Age", "Fare", "SibSp", "Parch"] if c in df.columns]:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_summary.append(f"{col}: {count} outlier(s), bounds [{lower:.3f}, {upper:.3f}]")
    notes.append("Outlier screening using the IQR rule (outliers retained for review):")
    notes.extend(outlier_summary if outlier_summary else ["No applicable numeric columns found for IQR screening."])

    return df, notes


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables and standardize selected numeric features."""
    processed = df.copy()

    # Identifier and free-text columns are not useful as raw numeric model features.
    # Keep them in the cleaned dataset, but omit them from this model-ready dataset.
    drop_cols = [col for col in ["Name", "Ticket", "Cabin", "PassengerId"] if col in processed.columns]
    processed = processed.drop(columns=drop_cols)

    # One-hot encode categories (drop_first avoids a redundant binary category).
    categorical_cols = processed.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        processed = pd.get_dummies(
            processed, columns=categorical_cols, drop_first=True, dtype=int
        )

    # Scale predictor columns only; do not scale the target label Survived.
    scale_cols = [
        col for col in ["Age", "Fare", "SibSp", "Parch"]
        if col in processed.columns and pd.api.types.is_numeric_dtype(processed[col])
    ]
    if scale_cols:
        scaler = StandardScaler()
        processed[scale_cols] = scaler.fit_transform(processed[scale_cols])

    return processed


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_df = load_data(DATA_PATH)

    initial_report = explore_data(raw_df)
    cleaned_df, cleaning_notes = clean_data(raw_df)
    processed_df = preprocess_data(cleaned_df)

    cleaned_path = OUTPUT_DIR / "cleaned_titanic.csv"
    processed_path = OUTPUT_DIR / "preprocessed_titanic.csv"
    summary_path = OUTPUT_DIR / "cleaning_summary.txt"

    cleaned_df.to_csv(cleaned_path, index=False)
    processed_df.to_csv(processed_path, index=False)

    final_report = [
        initial_report,
        "",
        "CLEANING AND PREPROCESSING ACTIONS",
        "=" * 58,
        *[f"- {note}" for note in cleaning_notes],
        "",
        f"Rows after cleaning: {cleaned_df.shape[0]}",
        f"Columns after cleaning: {cleaned_df.shape[1]}",
        f"Remaining missing values after cleaning: {int(cleaned_df.isna().sum().sum())}",
        f"Preprocessed dataset shape: {processed_df.shape}",
        f"Remaining missing values after preprocessing: {int(processed_df.isna().sum().sum())}",
        "",
        "Interpretation and limitations:",
        "- Median imputation preserves rows but can reduce variability and may bias distributions.",
        "- Mode imputation preserves valid categories but can over-represent the most common category.",
        "- IQR flags are diagnostic only; they are not automatically removed because unusual fares or family sizes may be genuine.",
        "- One-hot encoding makes categorical variables usable by many machine-learning algorithms.",
        "- Standardization places selected numeric predictors on comparable scales; the target Survived is not scaled.",
        "- Fit the scaler on training data only when building a predictive model to avoid data leakage.",
        "",
        "Output files:",
        str(cleaned_path),
        str(processed_path),
    ]
    summary_path.write_text("\n".join(final_report), encoding="utf-8")

    print("Data cleaning and preprocessing completed.")
    print(f"Input rows/columns: {raw_df.shape}")
    print(f"Cleaned rows/columns: {cleaned_df.shape}")
    print(f"Preprocessed rows/columns: {processed_df.shape}")
    print(f"Summary: {summary_path}")
    print(f"Cleaned CSV: {cleaned_path}")
    print(f"Preprocessed CSV: {processed_path}")


if __name__ == "__main__":
    main()
