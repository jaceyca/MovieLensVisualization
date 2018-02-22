import numpy as np
from surprise import Dataset, evaluate, Reader, prediction_algorithms, accuracy
from surprise.prediction_algorithms import knns, matrix_factorization, slope_one, co_clustering
from surprise.model_selection import cross_validate, GridSearchCV


def gridSearch(algo, param_grid, data):
    '''
    Runs grid search on a given algorithm with a given param_grid

    Input:
        algo: the algorithm we want to tune parameters for
        param_grid: the values we want to try
        data: test data for fitting
    '''
    gs = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    # best RMSE score
    print(gs.best_score['rmse'])
    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

reader = Reader(line_format='user item rating', sep='\t')
train_data = Dataset.load_from_file('./data/train.txt', reader=reader)
Y_train = train_data.build_full_trainset()

test_data = Dataset.load_from_file('./data/test.txt', reader=reader)
Y_test = test_data.build_full_trainset()
test_set = Y_test.build_testset()

# Y_test_str = np.loadtxt('./data/test.txt').astype(str)
# Y_test_int = np.loadtxt('./data/test.txt').astype(int)
# uid = list(Y_test_str[:, 0])
# mid = list(Y_test_str[:, 1])
# rating = list(Y_test_int[:, 2])

sim_options = {
    'name': 'cosine',
    'user_based': False
}

# Basic KNN algorithm
algo1 = knns.KNNBasic(k = 20, sim_options=sim_options)

# Means KNN algorithm
algo2 = knns.KNNWithMeans(k=20, sim_options=sim_options)

# ZScore KNN algorithm
algo3 = knns.KNNWithZScore(k = 20, sim_options=sim_options)

# Baseline KNN algorithm
algo4 = knns.KNNBaseline(k = 20, sim_options=sim_options)

# SVD
algo5 = matrix_factorization.SVD()

# SVD++
algo6 = matrix_factorization.SVDpp()

# Non-negative Matrix factorization
algo7 = matrix_factorization.NMF()

# Slope one
algo8 = slope_one.SlopeOne()

# Co-clustering
algo9 = co_clustering.CoClustering()


# error = 0
# for index, user in enumerate(uid): 

# 	prediction = knn.predict(user, mid[index], r_ui=rating[index], verbose=True)
# 	if float(np.around(prediction.est)) != float(rating[index]):
# 		error += 1

# testError = float(error)/len(uid)
# print(testError)

# Cross-validate
# cross_validate(knn, train_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

algos = [algo1,algo2,algo3,algo4,algo5,algo6,algo7,algo8,algo9]

# Fit and train, compute root mean square error
# for algo in algos:
# 	algo.fit(Y_train)
# 	predictions = algo.test(test_set)
# 	print(algo)
# 	accuracy.rmse(predictions) # Then compute RMSE


# Grid Search
param_grid_KNN = {'n_epochs': [5, 10, 20, 30, 40], 'k':[20, 30, 40, 50, 60, 70, 100],
                            'sim_options': {'name': ['msd', 'cosine'],
                            'user_based': [False]}}

# gridSearch(knns.KNNBasic, param_grid_KNN, train_data)
# KNNBasic: {'n_epochs': 5, 'k': 40, 'sim_options': {'name': 'msd', 'user_based': False}}
# Training RMSE: 0.991622355748

# gridSearch(knns.KNNWithMeans, param_grid_KNN, train_data)
# KNNWithMeans: {'n_epochs': 5, 'k': 50, 'sim_options': {'name': 'msd', 'user_based': False}}
# Training RMSE: 0.944528173188

# gridSearch(knns.KNNWithZScore, param_grid_KNN, train_data)
# KNNWithZScore: {'n_epochs': 5, 'k': 50, 'sim_options': {'name': 'msd', 'user_based': False}}
# Training RMSE: 0.946851952709



