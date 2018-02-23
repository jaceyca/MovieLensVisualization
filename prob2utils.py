import numpy as np


def running_avg(old, new, counter):
    return old + (new-old)/counter


def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return (1-reg*eta)*Ui + eta * Vj * (Yij - np.dot(Ui,Vj))     


def grad_U_advanced(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return (1-reg*eta)*Ui + eta * Vj * (Yij - mu - np.dot(Ui,Vj) - ai - bj)     


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return (1-reg*eta)*Vj + eta * Ui * (Yij - np.dot(Ui,Vj))


def grad_V_advanced(Vj, Yij, Ui, ai, bj, mu, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return (1-reg*eta)*Vj + eta * Ui * (Yij - mu - np.dot(Ui,Vj) - ai - bj)


def get_err_advanced(U, V, Y, mu, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    # We first compute the total squared squared error
    err = 0.0
    for (i,j,Yij) in Y:
        err += 0.5 *((Yij - mu) - np.dot(U[i-1], V[:,j-1]) - a[i-1] - b[j-1])**2
    # Add error penalty due to regularization if regularization
    # parameter is nonzero
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        a_frobenius_norm = np.linalg.norm(a)
        b_frobenius_norm = np.linalg.norm(b)

        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
        err += 0.5 * reg * (a_frobenius_norm ** 2)
        err += 0.5 * reg * (b_frobenius_norm ** 2)
    # Return the mean of the regularized error
    return err / float(len(Y))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    # We first compute the total squared squared error
    err = 0.0
    for (i,j,Yij) in Y:
        err += 0.5 *(Yij - np.dot(U[i-1], V[:,j-1]))**2
    # Add error penalty due to regularization if regularization
    # parameter is nonzero
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
    # Return the mean of the regularized error
    return err / float(len(Y))

def find_a(Y, mu, M):
    result = np.zeros(M)
    count = np.zeros(M)
    for index, data in enumerate(Y):
        count[data[0]-1] += 1
        result[data[0]-1] = running_avg(result[data[0]-1], data[2], count[data[0]-1])

    return result - mu

def find_b(Y, mu, N):
    result = np.zeros(N)
    count = np.zeros(N)
    for index, data in enumerate(Y):
        count[data[1]-1] += 1
        result[data[1]-1] = running_avg(result[data[1]-1], data[2], count[data[1]-1])

    return result - mu


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, mode='basic'):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    # Initialize U, V  
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((K,N)) - 0.5

    # Find the average of the data, mu
    YijVec = Y[:, 2]
    mu = np.mean(YijVec)
    aVec = find_a(Y, mu, M)
    bVec = find_b(Y, mu, N)

    size = Y.shape[0]
    delta = None
    indices = np.arange(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        if mode == 'basic':
            before_E_in = get_err(U, V, Y, reg)
        else: 
            before_E_in = get_err_advanced(U, V, Y, mu, aVec, bVec, reg)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            # Update U[i], V[j]
            if mode == 'basic':
                U[i-1] = grad_U(U[i-1], Yij, V[:,j-1], reg, eta)
                V[:,j-1] = grad_V(V[:,j-1], Yij, U[i-1], reg, eta);
            if mode == 'advanced':
                U[i-1] = grad_U_advanced(U[i-1], Yij, V[:,j-1], aVec[i-1], bVec[j-1], mu, reg, eta)
                V[:,j-1] = grad_V_advanced(V[:,j-1], Yij, U[i-1], aVec[i-1], bVec[j-1], mu, reg, eta)

        # At end of epoch, print E_in
        if mode == 'basic':
            E_in = get_err(U, V, Y, reg)
        else:
            E_in = get_err_advanced(U, V, Y, mu, aVec, bVec, reg)
        print("Epoch %s, E_in (regularized MSE): %s"%(epoch + 1, E_in))

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break

    if mode == 'basic':
        unregMSE = get_err(U, V, Y)
    if mode == 'advanced':
        unregMSE = get_err_advanced(U, V, Y, mu, aVec, bVec)

    if mode == 'advanced':
        return (U, V, unregMSE, aVec, bVec, mu)
    else: 
        return(U, V, unregMSE)
