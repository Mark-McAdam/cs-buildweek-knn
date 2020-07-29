from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
train['cluster'] = knn.fit(train[['longitude']], train[['latitude']])
test['cluster'] = knn.predict(test[['longitude', 'latitude']])
px.scatter(train, x='longitude', y='latitude', color='cluster')
