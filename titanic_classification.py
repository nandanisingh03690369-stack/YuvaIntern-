"""Week 5 - Logistic Regression (classification) on the Titanic dataset."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, roc_auc_score)

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("../data/titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "Fare", "FamilySize"] + [c for c in df.columns if c.startswith("Embarked_")]
X, y = df[features], df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC :", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=150)
