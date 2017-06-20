# -*- coding: utf-8 -*-
'''
demoSMLR_20140714
'''


import numpy
import sklearn.svm
import matplotlib.pyplot

import smlr
import pylab


## Prepare classifier objects
svm = sklearn.svm.LinearSVC()
smlr = smlr.SMLR(max_iter=1000,tol=1e-5,verbose=1)

## Sample data generation

# Num of samples
N = 100

# Label vector
label4training = numpy.vstack((numpy.zeros((N, 1)), numpy.ones((N, 1))))
label4test = numpy.vstack((numpy.zeros((N, 1)), numpy.ones((N, 1))))

# Features
feature4class1 = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
feature4class2 = numpy.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

feature4training = numpy.vstack(((numpy.dot(numpy.ones((N, 1)), [feature4class1]),
                                  numpy.dot(numpy.ones((N, 1)), [feature4class2]))))
feature4test = numpy.vstack(((numpy.dot(numpy.ones((N, 1)), [feature4class1]),
                              numpy.dot(numpy.ones((N, 1)), [feature4class2]))))

#import numpy.random
numpy.random.seed(seed=1)

feature4training = feature4training \
                   + 0.5 * numpy.random.randn(feature4training.shape[0], feature4training.shape[1])
feature4test = feature4test \
               + 0.5 * numpy.random.randn(feature4test.shape[0], feature4test.shape[1])

# Scatter plot in the feature space

for n in range(len(label4training)):
    if label4training[n] == 0:
        matplotlib.pyplot.scatter(feature4training[n, 0], feature4training[n, 1], color='red')
    else:
        matplotlib.pyplot.scatter(feature4training[n, 0], feature4training[n, 1], color='blue')

matplotlib.pyplot.xlabel("Dimension 1")
matplotlib.pyplot.ylabel("Dimension 2")

pylab.xlim(-3, 3)
pylab.ylim(-3, 3)
matplotlib.pyplot.show()

# SMLR & SVM training
print "SMLR learning"
smlr.fit(feature4training, label4training)
print "SVM learning"
svm.fit(feature4training, label4training)

print "The SLMR weights obtained"
print numpy.transpose(smlr.coef_)

# Linear boundary in the feature space
for n in range(len(label4training)):
    if label4training[n] == 0:
        matplotlib.pyplot.scatter(feature4training[n, 0], feature4training[n, 1], color='red')
    else:
        matplotlib.pyplot.scatter(feature4training[n, 0], feature4training[n, 1], color='blue')

matplotlib.pyplot.xlabel("Dimension 1")
matplotlib.pyplot.ylabel("Dimension 2")
w = smlr.coef_[0, :]
x = numpy.arange(-5, 5, 0.001)
y = (-w[-1] - x * w[0]) / w[1]
matplotlib.pyplot.plot(x, y, color='black')
pylab.xlim(-3, 3)
pylab.ylim(-3, 3)
matplotlib.pyplot.show()

#generalization test
predictedLabelBySVM = svm.predict(feature4test)
predictedLabelBySMLR = smlr.predict(feature4test)

num_correct = 0
for n in range(len(label4test)):
    if label4test[n] == predictedLabelBySVM[n]:
        num_correct = num_correct + 1

svm_accuracy = numpy.double(num_correct) / len(label4test) * 100

num_correct = 0

for n in range(len(label4test)):
    if label4test[n] == predictedLabelBySMLR[n]:
        num_correct = num_correct + 1

smlr_accuracy = numpy.double(num_correct) / len(label4test) * 100

print "SVM accuracy: %s" % (svm_accuracy)
print "SMLR accuracy: %s" % (smlr_accuracy)
