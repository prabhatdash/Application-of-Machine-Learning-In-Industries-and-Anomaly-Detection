#avg. depth of tree for a specific sample.
from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1)
X = [[11], [12], [3], [15], [100], [11], [15]]
isolation_forest.fit(X)
anomaly_scores = isolation_forest.decision_function(X)
print("Anomaly Scores:")
for i, score in enumerate(anomaly_scores):
    print("Over No",i+1,": ",score)
