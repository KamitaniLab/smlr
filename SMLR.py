"""
SMLR (sparse multinomial logistic regression)
"""


import numpy
import SMLRupdate
from sklearn.base import BaseEstimator, ClassifierMixin
class SMLR(BaseEstimator, ClassifierMixin):
    """Sparce Multinomial Logistic Regression (SMLR) classifier.
	The API of this function is compatible with the logistic regression in scikit-learn.

    Parameters
    ----------
    n_iter: The number of iterations in training (default 100). 
    
    verbose: If 1, print verbose information (default).

    Attributes
    ----------
    `coef_` : array, shape = [n_classes, n_features]
        Coefficient of the features in the decision function.

    `intercept_` : array, shape = [n_classes]
        Intercept (a.k.a. bias) added to the decision function.


    References:
	Sparse estimation automatically selects voxels relevant for the decoding of fMRI activity patterns.
	Yamashita O, Sato MA, Yoshioka T, Tong F, Kamitani Y.
	Neuroimage. 2008.
	doi: 10.1016/j.neuroimage.2008.05.050.    
	
	
    """



    def __init__(self, n_iter=1000,verbose=1):
	self.n_iter=n_iter
	self.verbose = verbose
	#self.densify

        print "SMLR (sparse multinomial logistic regression)"
    
    def fit(self,feature,label):
        """fit(self, feature, label) method of SMLR instance
        Fit the model according to the given training data (in the same way as logistic.py in sklearn).
    
    	Parameters
    	----------
    	feature : array-like, shape = [n_samples, n_features]
        	Training vector, where n_samples in the number of samples and
        	n_features is the number of features.
    
    	label : array-like, shape = [n_samples]
        	Target vector for "feature"
    
    	Returns
    	-------
    	self : object
        Returns self.
        """

        #feature: matrix, whose size is # of samples by # of dimensions.
        #label: label vector, whose size is # of samples.
        #       If you treat a classification problem with C classes, please use 0,1,2,...,(C-1) to indicate classes
        
        
        #Check # of features, # of dimensions, and # of classes
        self.classes_, indices = numpy.unique(label,return_inverse=True)
        N=feature.shape[0]
        D=feature.shape[1]
        #C=numpy.max(label)+1
        #C=C.astype(int)
        C=len(self.classes_)
        
        #transoform label into a 1-d array to avoid possible errors
        label=indices
        
        
        #make class label based on 1-of-K representation
        label_1ofK=numpy.zeros((N,C))
        for n in range(N):
            label_1ofK[n,label[n]]=1
    
    
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        D=D+1
    
        
        #set initial values of theta (wieghts) and alpha (relavence parameters)
        theta=numpy.zeros((D,C))
        alpha=numpy.ones((D,C))
        isEffective=numpy.ones((D,C))
        effectiveFeature=range(D)
        
        #Variational baysian method (see Yamashita et al., 2008)
        for iteration in range(self.n_iter):
            
            #theta-step
            newThetaParam=SMLRupdate.thetaStep(theta,alpha,label_1ofK,feature,isEffective)
            theta=newThetaParam['mu']#the posterior mean of theta
            
            #alpha-step
            alpha=SMLRupdate.alphaStep(alpha,newThetaParam['mu'],newThetaParam['var'],isEffective)
            
            #pruning of irrelevant dimensions (that have large alpha values)
            isEffective=numpy.ones(theta.shape)
            isEffective[alpha>10**3]=0
            theta[alpha>10**3]=0
            
            dim_excluded=(numpy.all(isEffective==0,axis=1))
            dim_excluded=[d for d in range(len(dim_excluded)) if dim_excluded[d]]
            theta=numpy.delete(theta,dim_excluded,axis=0)
            alpha=numpy.delete(alpha,dim_excluded,axis=0)
            feature=numpy.delete(feature,dim_excluded,axis=1)
            isEffective=numpy.delete(isEffective,dim_excluded,axis=0)
            effectiveFeature=numpy.delete(effectiveFeature,dim_excluded,axis=0)
            
            #show progress
            if self.verbose:
                if (iteration+1)%numpy.round(self.n_iter*0.2) ==0 or iteration == 0:
                    num_effectiveWeights=numpy.sum(isEffective)
                    print "# of iterations: %d ,  # of effective dimensions: %d" %(iteration+1, len(effectiveFeature))
    
        temporal_theta=numpy.zeros((D,C))
        temporal_theta[effectiveFeature,:]=theta
        theta=temporal_theta
                    
        self.coef_=numpy.transpose(theta[:-1,:])
        self.intercept_=theta[-1,:]
        return theta

    def predict(self,feature):
        """predict(self, feature) method of SMLR instance
 	   Predict class labels for samples in feature (in the same way as logistic.py in sklearn).
    
    	Parameters
    	----------
    	feature : {array-like, sparse matrix}, shape = [n_samples, n_features]
        	Samples.
    
    	Returns
    	-------
    	C : array, shape = [n_samples]
        	Predicted class label per sample.
	"""
        N=feature.shape[0]
        D=feature.shape[1]
        
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        
        #load weights
        w=numpy.vstack((numpy.transpose(self.coef_),self.intercept_))
        C=w.shape[1]
        
        #predictive probability calculation
        p=numpy.zeros((N,C))
        predicted_label=list([])
        for n in range(N):
            p[n,:]=numpy.exp(numpy.dot(feature[n,:],w))
            p[n,:]=p[n,:]/sum(p[n,:])
            predicted_label.append(self.classes_[numpy.argmax(p[n,:])])
        return predicted_label

#    def decision_function(self, feature):
#        N=feature.shape[0]
#        D=feature.shape[1]
#
#        #add a bias term to feature
#        feature=numpy.hstack((feature,numpy.ones((N,1))))
#
#        #load weights
#        w=numpy.vstack((numpy.transpose(self.coef_),self.intercept_))
#        C=w.shape[1]
#
#        #predictive probability calculation
#        decisionValue=numpy.zeros((N,C))
#        for n in range(N):
#            decisionValue[n,:]=numpy.dot(feature[n,:],w)
#	
#	return decisionValue

    def predict_proba(self,feature):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes (in the same way as logistic.py in sklearn).

        Parameters
        ----------
        feature : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        N=feature.shape[0]
        D=feature.shape[1]
        
        #add a bias term to feature
        feature=numpy.hstack((feature,numpy.ones((N,1))))
        
        #load weights
        w=numpy.vstack((numpy.transpose(self.coef_),self.intercept_))
        C=w.shape[1]
        
        #predictive probability calculation
        p=numpy.zeros((N,C))
        predicted_label=numpy.zeros(N)
        for n in range(N):
            p[n,:]=numpy.exp(numpy.dot(feature[n,:],w))
            p[n,:]=p[n,:]/sum(p[n,:])
        return p

    def predict_log_proba(self,feature):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes (in the same way as logistic.py in sklearn).

        Parameters
        ----------
        feature : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

	p = self.predict_proba(feature)
	return numpy.log(p)



