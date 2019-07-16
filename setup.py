from setuptools import setup

setup(name='smlr',
      version='1.1.1',
      description='Python package for Sparse Multinomial Logistic Regression',
      author='Kei Majima',
      author_email='brainliner-admin@atr.jp',
      url='https://github.com/KamitaniLab/SMLR',
      license='MIT',
      packages=['smlr'],
      install_requires=['numpy', 'scipy', 'scikit-learn'])
