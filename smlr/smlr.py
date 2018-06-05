"""
SMLR (sparse multinomial logistic regression)
"""

from __future__ import print_function

import numpy
import scipy
import scipy.optimize
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class SMLR(BaseEstimator, ClassifierMixin):
    """Sparce Multinomial Logistic Regression (SMLR) classifier.

    The API of this function is compatible with the logistic regression in
    scikit-learn.

    Parameters:
        max_iter: The maximum number of iterations in training
            (default 1000; int).
        n_iter: The number of iterations in training (default 100).
        verbose: If 1, print verbose information (default).

    Attributes:
        `coef_`: array, shape = [n_classes, n_features]
            Coefficient of the features in the decision function.
        `intercept_`: array, shape = [n_classes]
            Intercept (a.k.a. bias) added to the decision function.

    References:
        Sparse estimation automatically selects voxels relevant for the
        decoding of fMRI activity patterns.
        Yamashita O, Sato MA, Yoshioka T, Tong F, Kamitani Y.
        Neuroimage. 2008.
        doi: 10.1016/j.neuroimage.2008.05.050.

    """

    def __init__(self, max_iter=1000, tol=1e-5, verbose=1):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        # self.densify

        print("SMLR (sparse multinomial logistic regression)")

    def fit(self, feature, label):
        """fit(self, feature, label) method of SMLR instance

        Fit the model according to the given training data (in the same way as
        logistic.py in sklearn).

        Parameters:
            feature: array-like, shape = [n_samples, n_features]
                    Training vector, where n_samples in the number of samples
                    and n_features is the number of features.
            label: array-like, shape = [n_samples]
                    Target vector for "feature"

        Returns:
            self: object
            Returns self.
        """

        # feature: matrix, whose size is # of samples by # of dimensions.
        # label: label vector, whose size is # of samples.
        # If you treat a classification problem with C classes, please
        # use 0,1,2,...,(C-1) to indicate classes

        # Check # of features, # of dimensions, and # of classes
        self.classes_, indices = numpy.unique(label, return_inverse=True)
        N = feature.shape[0]
        D = feature.shape[1]
        # C=numpy.max(label)+1
        # C=C.astype(int)
        C = len(self.classes_)

        # transoform label into a 1-d array to avoid possible errors
        label = indices

        # make class label based on 1-of-K representation
        label_1ofK = numpy.zeros((N, C))
        for n in range(N):
            label_1ofK[n, label[n]] = 1

        # add a bias term to feature
        feature = numpy.hstack((feature, numpy.ones((N, 1))))
        D += 1

        # set initial values of theta (wieghts) and
        # alpha (relavence parameters)
        theta = numpy.zeros((D, C))
        alpha = numpy.ones((D, C))
        isEffective = numpy.ones((D, C))
        effectiveFeature = range(D)
        num_effectiveWeights = numpy.sum(isEffective)

        # Variational baysian method (see Yamashita et al., 2008)
        for iteration in range(self.max_iter):

            # theta-step
            newThetaParam = self.__thetaStep(
                theta, alpha, label_1ofK, feature, isEffective)
            theta = newThetaParam['mu']  # the posterior mean of theta
            if iteration == 0:
                funcValue_pre = newThetaParam['funcValue']
                funcValue = newThetaParam['funcValue']
            else:
                funcValue_pre = funcValue
                funcValue = newThetaParam['funcValue']

            # alpha-step
            alpha = self.__alphaStep(
                alpha, newThetaParam['mu'], newThetaParam['var'], isEffective)

            # pruning of irrelevant dimensions (that have large alpha values)
            isEffective = numpy.ones(theta.shape)
            isEffective[alpha > 1e+3] = 0
            theta[alpha > 1e+3] = 0

            dim_excluded = numpy.where(numpy.all(isEffective == 0, axis=1))
            theta = numpy.delete(theta, dim_excluded, axis=0)
            alpha = numpy.delete(alpha, dim_excluded, axis=0)
            feature = numpy.delete(feature, dim_excluded, axis=1)
            isEffective = numpy.delete(isEffective, dim_excluded, axis=0)
            effectiveFeature = numpy.delete(
                effectiveFeature, dim_excluded, axis=0)

            # show progress
            if self.verbose:
                if not num_effectiveWeights == numpy.sum(isEffective):
                    num_effectiveWeights = numpy.sum(isEffective)
                    print("# of iterations: %d,  # of effective dimensions: %d"
                          % (iteration + 1, len(effectiveFeature)))
                    print("# of iterations: %d,  FuncValue: %f"
                          % (iteration + 1, newThetaParam['funcValue']))
            if iteration > 1 and abs(funcValue - funcValue_pre) < self.tol:
                break

        temporal_theta = numpy.zeros((D, C))
        temporal_theta[effectiveFeature, :] = theta
        theta = temporal_theta

        self.coef_ = numpy.transpose(theta[:-1, :])
        self.intercept_ = theta[-1, :]
        return self

    def predict(self, feature):
        """predict(self, feature) method of SMLR instance

        Predict class labels for samples in feature (in the same way as
        logistic.py in sklearn).

        Parameters:
            feature: {array-like, sparse matrix},
                shape = [n_samples, n_features]
                Samples.

        Returns:
            C: array, shape = [n_samples]
                Predicted class label per sample.
        """

        # add a bias term to feature
        feature = numpy.hstack((feature, numpy.ones((feature.shape[0], 1))))

        # load weights
        w = numpy.vstack((numpy.transpose(self.coef_), self.intercept_))

        # predictive probability calculation
        p = numpy.exp(feature.dot(w))
        p /= p.sum(axis=1)[:, numpy.newaxis]
        predicted_label = self.classes_[numpy.argmax(p, axis=1)]
        return numpy.array(predicted_label)

    def decision_function(self, feature):
        # add a bias term to feature
        feature = numpy.hstack((feature, numpy.ones((feature.shape[0], 1))))

        # load weights
        w = numpy.vstack((numpy.transpose(self.coef_), self.intercept_))

        return feature.dot(w)

    def predict_proba(self, feature):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes (in the same way as logistic.py in sklearn).

        Parameters:
            feature: array-like, shape = [n_samples, n_features]

        Returns:
            T: array-like, shape = [n_samples, n_classes]
                Returns the probability of the sample for each class
                in the model, where classes are ordered as they are in
                ``self.classes_``.
        """

        # add a bias term to feature
        feature = numpy.hstack((feature, numpy.ones((feature.shape[0], 1))))

        # load weights
        w = numpy.vstack((numpy.transpose(self.coef_), self.intercept_))

        # predictive probability calculation
        p = numpy.exp(feature.dot(w))
        p /= p.sum(axis=1)[:, numpy.newaxis]
        return p

    def predict_log_proba(self, feature):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes (in the same way as logistic.py in sklearn).

        Parameters:
            feature: array-like, shape = [n_samples, n_features]

        Returns:
            T: array-like, shape = [n_samples, n_classes]
                Returns the log-probability of the sample for each class
                in the model, where classes are ordered as they are in
                ``self.classes_``.
        """
        p = self.predict_proba(feature)
        return numpy.log(p)

    def __thetaStep(self, theta, alpha, Y, X, isEffective):
        # chack # of dimensions, # of samples, and # of classes

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
            return -self.__funcE(theta_originalShape, alpha, Y, X)

        # set the gradient for Newton-CG based optimization
        def grad2minimize(theta_concatenated):
            theta_originalShape = thetaConcatenated2thetaOriginalShape(
                theta_concatenated)
            gradE_originalShape = self.__gradE(
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
            HessE_originalShape = self.__HessE(
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

        param = {'mu': mu, 'var': var, 'funcValue': res['fun']}
        return param

    def __alphaStep(self, alpha, theta, var, isEffective):
        D = alpha.shape[0]
        C = alpha.shape[1]
        for c in range(C):
            for d in range(D):
                if isEffective[d, c] == 1:
                    alpha[d, c] = (1 - alpha[d, c] * var[d, c]) / theta[d, c] ** 2
                else:
                    alpha[d, c] = 1e+8
        return alpha

    def __funcE(self, theta, alpha, Y, X):
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

    def __gradE(self, theta, alpha, Y, X):
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

    def __HessE(self, theta, alpha, Y, X):
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
