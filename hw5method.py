# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair


import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
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
    print(np.array(V).shape)
    print(np.array(U).shape)
    A, S, B = svd(V)
    A = np.array(A)
    A = A[:, [0, 1]]
    print(A.shape)
    # newU = np.dot(np.transpose(A), U)
    newV = np.dot(np.transpose(A), V)

    return newV



def main():
    Y_train = np.loadtxt('./data/train.txt').astype(int)
    Y_test = np.loadtxt('./data/test.txt')  .astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    Ks = [20]

    # this gives me the best results, but I could be wrong
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
            U,V, e_in = train_model(M, N, k, eta, reg, Y_train)
            E_ins_for_lambda.append(e_in)
            eout = get_err(U, V, Y_test)
            E_outs_for_lambda.append(eout)
            
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)


    newU, newV = projectUV(U, V)
    print("V:")
    print(newV)



if __name__ == "__main__":
    main()

   


