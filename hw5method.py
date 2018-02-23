import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err, get_err_advanced
from visualization import get_rating_freq, bar_plot, fancy_plot
from numpy.linalg import svd
import pylab as pl
import codecs
import operator
from collections import Counter



def projectUV(U, V):
    '''
    This function projects U and V in to a 2D space so we can create visualizations.

    Input: 
        U: The U matrix from SVD
        V: The V matrix from SVD

    Output: 
        newU: The 2D version of U
        newV: The 2D version of V
    '''

    A, S, B = svd(V)
    A = np.array(A)
    A = A[:, [0, 1]]
    newU = np.dot(np.transpose(A), np.transpose(U))
    newV = np.dot(np.transpose(A), V)

    return newU, newV


def main():
    Y_train = np.loadtxt('./data/train.txt').astype(int)
    Y_test = np.loadtxt('./data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    Ks = [20]

    # Ein and Eout for different regs have been recorded
    regs = [10**-1]
    eta = 0.03 # learning rate
    E_ins = []
    E_outs = []

    # Use to compute Ein and Eout
    for reg in regs:
        E_ins_for_lambda = []
        E_outs_for_lambda = []
        
        for k in Ks:
            print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, k, eta, reg))
            # U,V, e_in = train_model(M, N, k, eta, reg, Y_train, mode='basic')
            # E_ins_for_lambda.append(e_in)
            # eout = get_err(U, V, Y_test)

            U,V, e_in, aVec, bVec, mu = train_model(M, N, k, eta, reg, Y_train, mode='advanced')
            E_ins_for_lambda.append(e_in)
            eout = get_err_advanced(U, V, Y_test, mu, aVec, bVec, reg)

            E_outs_for_lambda.append(eout)
            
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)

    # basic gives Ein = 0.322 and Eout = 0.4565
    # advanced gives Ein = and Eout = FIX ERRORS AND SHIT


    newU, newV = projectUV(U, V)
    preds = np.dot(np.transpose(newU), newV)

    for index, row in enumerate(preds): 
        row[:] = row + aVec[index] + mu
    for index, column in enumerate(np.transpose(preds)):
        column[:] = column + bVec[index]
    intPreds = np.around(preds).astype(int)
    flatPreds = list(np.ndarray.flatten(intPreds))

    print('Errors:')
    print(E_ins)
    print(E_outs)

    r1 = flatPreds.count(1)
    r2 = flatPreds.count(2)
    r3 = flatPreds.count(3)
    r4 = flatPreds.count(4)
    r5 = flatPreds.count(5)
    f1 = r1/(r1+r2+r3+r4+r5)
    f2 = r2/(r1+r2+r3+r4+r5)
    f3 = r3/(r1+r2+r3+r4+r5)
    f4 = r4/(r1+r2+r3+r4+r5)
    f5 = r5/(r1+r2+r3+r4+r5)

    rating_freq = [f1, f2, f3, f4, f5]
    bar_plot(rating_freq, "Advanced Ratings of All Predicted Movies")


    # Plotting for actual things
    data = np.loadtxt('./data/data.txt').astype(int)
    movie_file = codecs.open('./data/movies.txt', mode='r', encoding='windows-1252')
    movie_names = {}
    genres = {}
    for line in movie_file:
        movie_info = line.split()
        movie_names[int(movie_info[0])] = " ".join(movie_info[1:-19])
        genres[int(movie_info[0])] = list(map(int, movie_info[-19:]))

    frequencies = Counter(data[:,1]) # how often the movies are reviewed
    avg_ratings = {}
    for data_tuple in data:
        key = data_tuple[1]
        avg_ratings[key] = avg_ratings.get(key, 0) + data_tuple[2]/frequencies[key]

    pop_movie_ids = [50, 258, 100, 181, 294, 286, 288, 1, 300, 121]
    pop_movie_names = ['Star Wars (1977)', 'Contact (1997)', 'Fargo (1996)', 'Return of the Jedi (1983)', 
    'Liar Liar (1997)', '"English Patient, The (1996)"', 'Scream (1996)', 'Toy Story (1995)', 
    'Air Force One (1997)', 'Independence Day (ID4) (1996)']
    pop_rating_ratings = np.ndarray.flatten(intPreds[:, pop_movie_ids])
    pop_freqs_count = Counter(pop_rating_ratings)
    pop_f1 = pop_freqs_count[1]/float(len(pop_rating_ratings))
    pop_f2 = pop_freqs_count[2]/float(len(pop_rating_ratings))
    pop_f3 = pop_freqs_count[3]/float(len(pop_rating_ratings))
    pop_f4 = pop_freqs_count[4]/float(len(pop_rating_ratings))
    pop_f5 = pop_freqs_count[5]/float(len(pop_rating_ratings))
    pop_freqs = [pop_f1, pop_f2, pop_f3, pop_f4, pop_f5]

    bar_plot(pop_freqs, "Advanced Predictions of Top Ten Popular Movies")

if __name__ == "__main__":
    main()

   


