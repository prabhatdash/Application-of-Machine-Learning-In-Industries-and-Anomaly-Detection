from sklearn.neighbors import LocalOutlierFactor
X = [[10], [10.1], [10.2], [10.3], [2.5]]
lof = LocalOutlierFactor(n_neighbors=2)
outlier_scores = lof.fit_predict(X)
outliers = list((zip(X,outlier_scores)))

for i in outliers:
    if i[1]==-1:
        print("OUTLIER IS:",i)