import numpy as np
from surprise import Dataset, evaluate, KNNBasic, Reader


# for data in Y_train: 

reader = Reader(line_format='user item rating', sep='\t')
Y_train = Dataset.load_from_file('./data/train.txt', reader = reader)
Y_train = Y_train.build_full_trainset()

Y_test = np.loadtxt('./data/test.txt').astype(str)
uid = list(Y_test[:, 0])
mid = list(Y_test[:, 1])
rating = list(Y_test[:, 2])

sim_options = {
    'name': 'cosine',
    'user_based': False
}

knn = KNNBasic(k = 20, sim_options=sim_options)
knn.fit(Y_train)

error = 0
for index, user in enumerate(uid): 
	prediction = knn.predict(user, mid[index])
	print(prediction)
	if float(np.around(prediction.est)) != float(rating[index]):
		error += 1
testError = float(error)/len(uid)
print(testError)