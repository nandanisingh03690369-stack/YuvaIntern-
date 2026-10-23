"""Week 6 - K-Means Clustering on the Iris dataset."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("../data/iris.csv").drop(columns=["Id"], errors="ignore")
X = StandardScaler().fit_transform(df.drop(columns=["Species"]))

# Elbow + silhouette to choose k
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig("outputs/elbow_silhouette.png", dpi=150)

# Final model (k=3 -> three iris species)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
df["Cluster"] = km.labels_
print("Silhouette (k=3):", silhouette_score(X, km.labels_))
print(pd.crosstab(df["Species"], df["Cluster"]))

pts = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6, 5))
plt.scatter(pts[:, 0], pts[:, 1], c=km.labels_, cmap="viridis")
plt.title("K-Means Clusters (PCA 2D)")
plt.savefig("outputs/clusters_pca.png", dpi=150)
