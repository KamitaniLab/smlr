# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy
import scipy
import scipy.optimize
from smlr import SMLRsubfunc


def thetaStep(theta, alpha, Y, X, isEffective):

    # chack # of dimensions, # of samples, and # of classes
    N = X.shape[0]
    D = X.shape[1]
    C = Y.shape[1]

    # Take indices for effective features (if alpha > 10^3, that dimension is
    # ignored in the following optimization steps)
    FeatureNum_effectiveWeight = []
    ClassNum_effectiveWeight = []
    for c in range(C):
        for d in range(D):
            if isEffective[d, c] == 1:
                FeatureNum_effectiveWeight.append(d)
                ClassNum_effectiveWeight.append(c)

    # Declaration of subfunction. this function transform concatenated
    # effective weight paramters into the original shape
    def thetaConcatenated2thetaOriginalShape(theta_concatenated):
        if len(theta_concatenated) != len(FeatureNum_effectiveWeight):
            raise ValueError("The size of theta_concatenated is wrong")

        theta_original = numpy.zeros((D, C))
        for index_effective_weight in range(len(FeatureNum_effectiveWeight)):
            theta_original[
                FeatureNum_effectiveWeight[index_effective_weight],
                ClassNum_effectiveWeight[index_effective_weight]] =\
                theta_concatenated[index_effective_weight]
        return theta_original

    # set the cost function that will be minimized in the following 
    # optimization
    def func2minimize(theta_concatenated):
        theta_originalShape = thetaConcatenated2thetaOriginalShape(
            theta_concatenated)
        return -SMLRsubfunc.funcE(theta_originalShape, alpha, Y, X)

    # set the gradient for Newton-CG based optimization
    def grad2minimize(theta_concatenated):
        theta_originalShape = thetaConcatenated2thetaOriginalShape(
            theta_concatenated)
        gradE_originalShape = SMLRsubfunc.gradE(
            theta_originalShape, alpha, Y, X)

        # ignore the dimensions that have large alphas
        dim_ignored = isEffective.ravel(order='F')[:, numpy.newaxis]
        dim_ignored = numpy.nonzero(1 - dim_ignored)
        gradE_used = numpy.delete(gradE_originalShape, dim_ignored[0])
        return -gradE_used
    # set the Hessian for Newton-CG based optimization

    def Hess2minimize(theta_concatenated):
        theta_originalShape = thetaConcatenated2thetaOriginalShape(
            theta_concatenated)
        HessE_originalShape = SMLRsubfunc.HessE(
            theta_originalShape, alpha, Y, X)

        # ignore the dimensions that have large alphas
        dim_ignored = isEffective.ravel(order='F')[:, numpy.newaxis]
        dim_ignored = numpy.nonzero(1 - dim_ignored)
        HessE_used = numpy.delete(HessE_originalShape, dim_ignored[0], axis=0)
        HessE_used = numpy.delete(HessE_used, dim_ignored[0], axis=1)
        return -HessE_used

    # set the initial value for optimization. we use the current theta for 
    # this.
    x0 = theta.ravel(order='F')[:, numpy.newaxis]
    dim_ignored = isEffective.ravel(order='F')[:, numpy.newaxis]
    dim_ignored = numpy.nonzero(1 - dim_ignored)
    x0 = numpy.delete(x0, dim_ignored[0])

    # Optimization of theta (weight paramter) with scipy.optimize.minimize
    res = scipy.optimize.minimize(
        func2minimize, x0, method='Newton-CG',
        jac=grad2minimize, hess=Hess2minimize, tol=1e-3)
    mu = thetaConcatenated2thetaOriginalShape(res['x'])

    # The covariance matrix of the posterior distribution
    cov = numpy.linalg.inv(Hess2minimize(res['x']))

    # The diagonal elements of the above covariance matrix
    var = numpy.diag(cov)
    var = thetaConcatenated2thetaOriginalShape(var)

    param = {'mu': mu, 'var': var}
    return param
# <codecell>


# <codecell>

def alphaStep(alpha, theta, var, isEffective):
    # change the type of input values i

    effectiveIndices = numpy.where(isEffective[:, 0] == 1)
    newAlpha = numpy.ones_like(alpha) * 1e+8
    newAlpha[effectiveIndices] = (1 - alpha * var) / theta ** 2
    return newAlpha
