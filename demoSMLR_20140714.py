# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from numpy import *
import numpy

#import SMLR
from SMLR import *

#import SVM from scikir-learn
import sklearn.svm

#prepare classifier objects
svm=sklearn.svm.LinearSVC()
smlr=SMLR()

# <codecell>

#sample data generation

# of samples
N=100

#label vector
label4training=numpy.vstack((numpy.zeros((N,1)),numpy.ones((N,1))))
label4test=numpy.vstack((numpy.zeros((N,1)),numpy.ones((N,1))))

#features 
feature4class1=numpy.array([1,0,0,0,0,0,0,0,0,0])
feature4class2=numpy.array([-1,0,0,0,0,0,0,0,0,0])

feature4training=numpy.vstack(((numpy.dot(numpy.ones((N,1)),[feature4class1]),numpy.dot(numpy.ones((N,1)),[feature4class2]))))
feature4test=numpy.vstack(((numpy.dot(numpy.ones((N,1)),[feature4class1]),numpy.dot(numpy.ones((N,1)),[feature4class2]))))
import numpy.random
numpy.random.seed(seed=1)
feature4training=feature4training+0.5*numpy.random.randn(feature4training.shape[0],feature4training.shape[1])
feature4test=feature4test+0.5*numpy.random.randn(feature4test.shape[0],feature4test.shape[1])

# <codecell>

#scatter plot in the feature space
import matplotlib.pyplot
for n in range(len(label4training)):
    if label4training[n]==0:
        matplotlib.pyplot.scatter(feature4training[n,0],feature4training[n,1],color='red')
    else:
        matplotlib.pyplot.scatter(feature4training[n,0],feature4training[n,1],color='blue')
matplotlib.pyplot.xlabel("Dimension 1")
matplotlib.pyplot.ylabel("Dimension 2")
import pylab
pylab.xlim(-3,3)
pylab.ylim(-3,3)
matplotlib.pyplot.show()

# <codecell>

#SMLR & SVM training
print "SMLR learning"
smlr.fit(feature4training,label4training)
print "SVM learning"
svm.fit(feature4training,label4training)

# <codecell>

print "The SLMR weights obtained"
print numpy.transpose(smlr.coef_)

# <codecell>

#linear boundary in the feature space
import matplotlib.pyplot
for n in range(len(label4training)):
    if label4training[n]==0:
        matplotlib.pyplot.scatter(feature4training[n,0],feature4training[n,1],color='red')
    else:
        matplotlib.pyplot.scatter(feature4training[n,0],feature4training[n,1],color='blue')
matplotlib.pyplot.xlabel("Dimension 1")
matplotlib.pyplot.ylabel("Dimension 2")
w=smlr.coef_[0,:]
x = numpy.arange(-5, 5, 0.001)
y = (-w[-1]-x*w[0])/w[1]
matplotlib.pyplot.plot(x,y,color='black')
import pylab
pylab.xlim(-3,3)
pylab.ylim(-3,3)
matplotlib.pyplot.show()

# <codecell>

#generalization test
predictedLabelBySVM=svm.predict(feature4test)
predictedLabelBySMLR=smlr.predict(feature4test)

num_correct=0
for n in range(len(label4test)):
    if label4test[n]==predictedLabelBySVM[n]:
        num_correct=num_correct+1
SVMaccuracy=numpy.double(num_correct)/len(label4test)*100

num_correct=0
for n in range(len(label4test)):
    if label4test[n]==predictedLabelBySMLR[n]:
        num_correct=num_correct+1
SMLRaccuracy=numpy.double(num_correct)/len(label4test)*100

print "SVM accuracy: %s"%(SVMaccuracy)
print "SMLR accuracy: %s"%(SMLRaccuracy)

