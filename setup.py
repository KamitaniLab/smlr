from setuptools import setup

setup(name='smlr',
      version='1.0',
      description='Python package for Sparse Multinomial Logistic Regression',
      author='Kei Majima',
      author_email='kamitanilab.contact@gmail.com',
      url='https://github.com/KamitaniLab/SMLR',
      license='MIT',
      packages=['smlr'],
      install_requires=['numpy', 'scipy', 'scikit-learn'])
