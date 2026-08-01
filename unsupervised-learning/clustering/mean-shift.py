from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

iris = load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df.head()

X = iris_df[iris.feature_names].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=len(X_scaled))
print("Estimated Bandwidth:", bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_scaled)

labels = ms.labels_
centers_scaled = ms.cluster_centers_

print("Number of clusters found:", len(centers_scaled))

iris_df["cluster"] = labels

print(iris_df[["species", "cluster"]].head())
print("\nCluster vs Species:")
print(pd.crosstab(iris_df["cluster"], iris_df["species"]))

plt.figure(figsize=(7, 5))

plt.scatter(
    iris_df["petal length (cm)"],
    iris_df["petal width (cm)"],
    c=iris_df["cluster"],
    cmap="viridis",
    alpha=0.8
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Mean Shift Clustering")
plt.colorbar(label="Cluster ID")
plt.show()

