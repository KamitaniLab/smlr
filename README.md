# SMLR: Sparse Multinominal Logistic Regression

Sparse Multinomial Logistic Regression (SMLR) classifier, developed by Kei Majima at Kyoto Univ.
The API of this function is compatible with the logistic regression in scikit-learn.

Sparse regularization by automatic relevance determination (ARD) prior was introduced to the linear multinomial logistic regression algorithm (Yamashita et al., 2008).
This regularization process estimates the importance of each voxel (feature) and prunes away voxels that are not useful for prediction.

Original SLR toolbox for Matlab is available at <http://www.cns.atr.jp/%7Eoyamashi/SLR_WEB.html>.

## Installation

### 1. Install required packages

- numpy
- scipy
- scikit-learn

To run the sample script (`demoSMLR_20140714.py`), matplotlib is also required.

### 2. Run the setup script

```shell
$ python setup.py install
```

## Usage

``` python
import smlr

model = smlr.SMLR(n_iter=100)
model.fit(x, y)
model.predict(x_test)
```

- `x`, `x_text`: numpy array of input features (# of samples x # of features)
- `y`: label vector consisting of integers (len (y) = # of samples; please use integers 0, 1, 2, ..., K-1 when K-class classification)

### Parameters

- `n_iter`: The number of iterations in training (default `100`).
- `verbose`: If 1, print verbose information (default).

### Attributes

- `coef_`: array, shape = [n_classes, n_features]
    - Coefficient of the features in the decision function.
- `intercept_`: array, shape = [n_classes]
    - Intercept (a.k.a. bias) added to the decision function.

For demonstration, try `demoSMLR_20140714.py`.

## References

Yamashita O, Sato MA, Yoshioka T, Tong F, Kamitani Y. (2008) Sparse estimation automatically selects voxels relevant for the decoding of fMRI activity patterns. NeuroImage. doi: 10.1016/j.neuroimage.2008.05.050. <http://www.sciencedirect.com/science/article/pii/S1053811908006940>

## License

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).
