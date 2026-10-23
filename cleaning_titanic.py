"""Week 2 - Data Cleaning & Preprocessing on the Titanic dataset."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/titanic.csv")
print("Before cleaning:\n", df.isnull().sum())

# 1. Missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])  # ~77% missing -> dropped

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Outliers in Fare (IQR capping)
q1, q3 = df["Fare"].quantile([0.25, 0.75])
iqr = q3 - q1
df["Fare"] = df["Fare"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 4. Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 5. Encoding + scaling
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
df[["Age", "Fare"]] = StandardScaler().fit_transform(df[["Age", "Fare"]])

df = df.drop(columns=["Name", "Ticket", "PassengerId"])
print("\nAfter cleaning:\n", df.isnull().sum())
df.to_csv("titanic_clean.csv", index=False)
print("Saved titanic_clean.csv", df.shape)
