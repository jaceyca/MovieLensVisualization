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

    A, S, B = svd(V)
    A = np.array(A)
    A = A[:, [0, 1]]
    newU = np.dot(np.transpose(A), np.transpose(U))
    newV = np.dot(np.transpose(A), V)

    return newU, newV


def get_rating_freq(data, movieIDs):
    '''
    This function gets relative frequency of each rating given the data 
    we read in and somee iterable item containing all the ID's of the 
    movies we want to tally

    Input: 
        data: the rating data that we read in 
        movieIDs: the IDs of the movies we are considering for the tally

    Output: 
        a list of how the frequency of each rating, from 1 to 5
    '''
    r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
    for rating in data: 
        if rating[1] in movieIDs: 
            if rating[2] == 1: r1 += 1
            elif rating[2] == 2: r2 += 1
            elif rating[2] == 3: r3 += 1
            elif rating[2] == 4: r4 += 1
            elif rating[2] == 5: r5 += 1
    f1 = r1/(r1+r2+r3+r4+r5)
    f2 = r2/(r1+r2+r3+r4+r5)
    f3 = r3/(r1+r2+r3+r4+r5)
    f4 = r4/(r1+r2+r3+r4+r5)
    f5 = r5/(r1+r2+r3+r4+r5)
    return [f1, f2, f3, f4, f5]

def bar_plot(rating_count, title):
    '''
    This function plots the rating count given in a bar graph

    Input: 
        rating_count: a list containing the number of each rating
        title: what the title of the plot should be
    Output: 
        shows the plot
        saves the plot under the title given 
    '''
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    width = 0.35  
    ind = np.arange(5) # x locations for the ratings

    rectangles = ax.bar(ind, rating_count, width, color='black')

    ax.set_xlim(-width,len(ind)-width)
    ax.set_ylim(0,max(rating_count))
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Movies')
    ax.set_title(title)
    xTickMarks = [str(i) for i in range(1,6)]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=10)
    plt.savefig(title)
    plt.show()


def main():
    Y_train = np.loadtxt('./data/train.txt').astype(int)
    Y_test = np.loadtxt('./data/test.txt')  .astype(int)

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
            U,V, e_in = train_model(M, N, k, eta, reg, Y_train)
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

   


