# PCA
Exploring PCA applications, for a bit of fun

# Overview
Princple Component Analysis (PCA) is a technique used to help interpret large data sets, by reducing the dimensionality, while at the same time miniming information loss.
It does this by creating new uncorrelated variables that successively maximize variance.
Finding the new variables, the principle components, reduces to solving an eigenvalue/eigenvector problem. 
The new variables are defined by the dataset the technique is being applied to, not a priori, thus classifying PCA as an adaptive data analysis technique (or unsupervised).

PCA can be based on either the covariance matrix or the correlation matrix.

The main use of PCA are for descriptive, rather than inferential purposes, there is no need for any distributional assumptions, and is an adaptive.
So it is a good tool for exploratory work on large data sets.

There is still a need to scale the data set, typically all 'features' are set to have a mean of zero

## Anomaly detection

For anomaly detection, each new input is analyzed, and the anomaly detection algorithm computes its projection onto the eigenvectors, together with a normalized reconstruction error. The normalized error is then used as the anomaly score. The higher the error, the more anomalous the instance is.


# Lay out of Repository

cmake  - modules to help cmake find dependencies
docs   - randomn notes I'm putting together
src
  cpp  - cpp source code
  python - python code for an overview of the programs see: https://github.com/mikemoorester/PCA/wiki/Python-Programs 

# Acknowledgements

Source code has been adapted and modified from https://github.com/ihar/EigenPCA

# Dependencies

Eigen 3 is used (http://eigen.tuxfamily.org/index.php?title=Main_Page)
Openblas

# Compiling

cd src
mkdir build
cd build
cmake ..

or
g++ -I/usr/include/eigen3  pca.cpp pca_run.cpp -o pca
./pca this will then read the test data and do stuff..

 
