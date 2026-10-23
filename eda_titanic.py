"""Week 1 - Exploratory Data Analysis on the Titanic dataset."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("../data/titanic.csv")

print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all"))
print("\nMissing values:\n", df.isnull().sum())
print("\nSurvival rate by Sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="Survived", ax=ax[0]); ax[0].set_title("Survival Count")
sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax[1]); ax[1].set_title("Survival by Class")
sns.histplot(df["Age"].dropna(), kde=True, ax=ax[2]); ax[2].set_title("Age Distribution")
plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=150)

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
