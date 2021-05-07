
# Overview of Principal component analysis 
Principal Component Analysis (PCA) is a technique used to help interpret large data sets, by reducing the dimensionality, while at the same time miniming information loss.
It does this by creating new uncorrelated variables that successively maximize variance.
Finding the new variables, the principle components, reduces to solving an eigenvalue/eigenvector problem. 
The new variables are defined by the dataset the technique is being applied to, not a priori, thus classifying PCA as an adaptive data analysis technique (or unsupervised).

PCA can be based on either the covariance matrix or the correlation matrix.

The main use of PCA are for descriptive, rather than inferential purposes.
PCA needs no distributional assumptions, and is an adaptive exploratory method which can be used on numerical data of various types.

# Key issues 
The properties of PCA have undesirable features when the variables have different units of measurement.
PCA is defined by a criterion (variance) that depends on units of measurement implies that PCs based on the covariance matrix S will change if th euntis of measurements on one or more of the variables change.
To overcome this it is common practise to begin by standardizing the variables.
Each data value x_ii is both centres and divided by the standard deviation s_j of the n observations of variable j,

z_ij = x_ij -x_j / s_j

So the initial data matrix X is replaced with the standardized data matrix Z, whose jth, is vector z_ with the n standardized observation of variable j.

Since the covariance matrix of a standardized dataset is just the correlation matrix R of the original dataset, a PCA on the standardized data is also known as a correlation matrix PCA. 
The eigenvectors a_k of the correlation matrix R define the uncorrelated maximum-variance linear combinations.


# References

Principal component analysis: a review and recent developments Ian Joliffe and Jorge Cadima 2016 https://doi.org/10.1098/rsta.2015.0202
 
