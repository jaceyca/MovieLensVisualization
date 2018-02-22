# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair


import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
from visualization import get_rating_freq, bar_plot, fancy_plot
from numpy.linalg import svd


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
            U,V, e_in = train_model(M, N, k, eta, reg, Y_train, mode='advanced')
            E_ins_for_lambda.append(e_in)
            eout = get_err(U, V, Y_test)
            E_outs_for_lambda.append(eout)
            
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)



    newU, newV = projectUV(U, V)
    preds = np.dot(np.transpose(newU), newV)
    intPreds = list(np.around(np.ndarray.flatten(preds)).astype(int))

    print("Number of Points:")
    print(len(intPreds))

    r1 = intPreds.count(1)
    r2 = intPreds.count(2)
    r3 = intPreds.count(3)
    r4 = intPreds.count(4)
    r5 = intPreds.count(5)
    f1 = r1/(r1+r2+r3+r4+r5)
    f2 = r2/(r1+r2+r3+r4+r5)
    f3 = r3/(r1+r2+r3+r4+r5)
    f4 = r4/(r1+r2+r3+r4+r5)
    f5 = r5/(r1+r2+r3+r4+r5)

    rating_freq = [f1, f2, f3, f4, f5]
    bar_plot(rating_freq, "Ratings of All Predicted Movies")

if __name__ == "__main__":
    main()

   


