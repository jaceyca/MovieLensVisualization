import numpy as np
from surprise import Dataset, evaluate, Reader, prediction_algorithms 


# for data in Y_train: 

reader = Reader(line_format='user item rating', sep='\t')
Y_train = Dataset.load_from_file('./data/train.txt', reader=reader)
Y_train = Y_train.build_full_trainset()

Y_test_str = np.loadtxt('./data/test.txt').astype(str)
Y_test_int = np.loadtxt('./data/test.txt').astype(int)
uid = list(Y_test_str[:, 0])
mid = list(Y_test_str[:, 1])
rating = list(Y_test_int[:, 2])

sim_options = {
    'name': 'cosine',
    'user_based': False
}

## Basic KNN algorithm
# knn = KNNBasic(k = 20, sim_options=sim_options)
# knn.fit(Y_train)

# Means KNN algorithm
knn = prediction_algorithms.knns.KNNWithMeans(k=20, sim_options=sim_options)
knn.fit(Y_train)

## ZScore KNN algorithm
# knn = KNNWithZScore(k = 20, sim_options=sim_options)
# knn.fit(Y_train)

## Baseline KNN algorithm
# knn = KNNBaseline(k = 20, sim_options=sim_options)
# knn.fit(Y_train)

# 

error = 0
for index, user in enumerate(uid): 
	prediction = knn.predict(user, mid[index], r_ui=rating[index], verbose=True)
	if float(np.around(prediction.est)) != float(rating[index]):
		error += 1

testError = float(error)/len(uid)
print(testError)