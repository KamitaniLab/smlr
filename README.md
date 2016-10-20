# SMLR

Sparce Multinomial Logistic Regression (SMLR) classifier.
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
  Yamashita O, Sato MA, Yoshioka T, Tong F, Kamitani Y.   Neuroimage. 2008.
  doi: 10.1016/j.neuroimage.2008.05.050.   

  http://www.cns.atr.jp/%7Eoyamashi/SLR_WEB.html


License:
The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php) 

