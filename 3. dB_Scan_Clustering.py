import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('hr_data.csv')

features = dataset.iloc[0:, [2,3]].values

print(features.shape)


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


dbscan = DBSCAN(eps=0.3, min_samples=2)
dbscan.fit(scaled_features)


labels = dbscan.labels_


dataset['Cluster'] = labels
print(dataset['Cluster'].to_markdown())


plt.scatter(dataset['average_montly_hours'], dataset['number_project'], c=labels, cmap='viridis')
plt.xlabel('average_montly_hours')
plt.ylabel('number_project')
plt.title('DBSCAN Clustering')
plt.show()