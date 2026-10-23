"""Week 3 - Data Visualization on the Iris dataset."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("../data/iris.csv").drop(columns=["Id"], errors="ignore")
features = [c for c in df.columns if c != "Species"]

sns.pairplot(df, hue="Species")
plt.savefig("outputs/pairplot.png", dpi=150)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="Species", y=col, ax=ax)
    ax.set_title(f"{col} by Species")
plt.tight_layout()
plt.savefig("outputs/boxplots.png", dpi=150)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), features):
    sns.violinplot(data=df, x="Species", y=col, ax=ax)
plt.tight_layout()
plt.savefig("outputs/violinplots.png", dpi=150)

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="viridis")
plt.title("Iris Feature Correlation")
plt.tight_layout()
plt.savefig("outputs/heatmap.png", dpi=150)
