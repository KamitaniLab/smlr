# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy


def funcE(theta, alpha, Y, X):
    # theta: weights for classification, dimensions by # of classes
    # alpha: relavance parameters in ARD, dimensions by # of classes
    # Y: label matrix, 1-of-K representation, # of samples by # of classes
    # X: feature matrix, # of samples by # of dimensions
    # output: log likelihood function of theta (averaged over alpha)

    linearSum = X.dot(theta)
    fone = numpy.sum(Y * linearSum, axis=1) - \
        numpy.log(numpy.sum(numpy.exp(linearSum), axis=1))
    E = numpy.sum(fone) - (0.5) * numpy.sum(theta.ravel(order='F') *
                                            alpha.ravel(order='F') ** 2)
    return E


def gradE(theta, alpha, Y, X):
    # theta: weights for classification, dimensions by # of classes
    # alpha: relavance parameters in ARD, dimensions by # of classes
    # Y: label matrix, 1-of-K representation, # of samples by # of classes
    # X: feature matrix, # of samples by # of dimensions
    # output: The gragient of funcE. This is used for Q-step optimization.

    D = X.shape[1]
    C = Y.shape[1]

    dE = numpy.zeros((theta.shape[0] * theta.shape[1], 1))
    linearSumExponential = numpy.exp(X.dot(theta))
    p = linearSumExponential / numpy.sum(
        linearSumExponential, axis=1)[:, numpy.newaxis]

    for c in range(C):
        temporal_dE = numpy.sum(
            (Y[:, c] - p[:, c]) * X.T, axis=1)[:, numpy.newaxis]
        A = numpy.diag(alpha[:, c])
        temporal_dE -= numpy.transpose([numpy.dot(A, theta[:, c])])
        dE[c * D:((c + 1) * D), 0] = numpy.squeeze(temporal_dE)

    return numpy.squeeze(dE)


def HessE(theta, alpha, Y, X):
    # theta: weights for classification, dimensions by # of classes
    # alpha: relavance parameters in ARD, dimensions by # of classes
    # Y: label matrix, 1-of-K representation, # of samples by # of classes
    # X: feature matrix, # of samples by # of dimensions
    # output: The Hessian of funcE. This is used for Q-step optimization.

    N = X.shape[0]
    D = X.shape[1]
    C = Y.shape[1]

    linearSumExponential = numpy.exp(X.dot(theta))
    p = linearSumExponential / numpy.sum(
        linearSumExponential, axis=1)[:, numpy.newaxis]
    H = numpy.zeros((C * D, C * D))

    for n in range(N):
        M1 = numpy.diag(p[n, :])
        M2 = numpy.dot(numpy.transpose([p[n, :]]), [p[n, :]])
        M3 = numpy.dot(numpy.transpose([X[n, :]]), [X[n, :]])
        H = H - numpy.kron(M1 - M2, M3)
    H = H - numpy.diag(alpha.ravel(order='F'))
    # Derivation came from Yamashita et al., Nuroimage,2008., but
    # exactly speaking, the last term is modified.
    # The above is consistent with SLR toolbox (MATLAB-based implementation by
    # Yamashita-san) rather than the paper.
    return H
