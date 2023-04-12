import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('iot_telemetry_data.csv')
data = data.iloc[0:1000, [3, 8]]
print(data.to_markdown())
X = data.values

# standardize the data
X = StandardScaler().fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# plot the results
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

for cluster, color in zip(range(n_clusters), colors):
    mask = (labels == cluster)
    plt.scatter(X[mask, 0], X[mask, 1], c=color)

# plt.title(f"DBSCAN Clustering (n_clusters={n_clusters})")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
info = f"DBSCAN Clustering, No of Clusters: {n_clusters} )"
plt.title(info)
plt.show()