import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#
# Download and convert the MNIST data set
# see the program: mnist2csv.py (converts idx3 data to csv)

# Load the dataset

df = pd.read_csv('data/mnist_train.csv')
#print( df.head() )

# Plot a couple of images

pixel_colnames = df.columns[1:]

# Get all columns except the label column for the first image

image_values = df.loc[0, pixel_colnames].values

# Plot up the first two images just for fun:

plt.figure(figsize=(8,4))

for index in range(0, 2):

    plt.subplot(1, 2, 1 + index )
    image_values = df.loc[index, pixel_colnames].values
    image_label = df.loc[index, 'label']
    plt.imshow(image_values.reshape(28,28), cmap ='gray')
    plt.title('Label: ' + str(image_label), fontsize = 18)

# Split data into Training and Test Sets

X_train, X_test, y_train, y_test = train_test_split(df[pixel_colnames], df['label'], random_state=0)

# Standardize the data:
# PCA and logisitic regression are sensitive to the scale of your features. 
# set the data onto unit scale (mean = 0 and variance = 1) by using StandardScaler

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the training set and the test set.

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# for the plot at the bottom
scaledTrainImages = X_train.copy()

# Perform PCA
# n_components = .90 means that scikit-learn will choose the minimum number 
# of principal components such that 90% of the variance is retained.

pca = PCA(n_components = .90)

# Fit PCA on training set only

pca.fit(X_train)

# Apply the mapping (transform) to both the training set and the test set. 

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Logistic Regression 
# - If you turn PCA off, you obtain a very similar accuracy as to using the
#   complete data set

clf = LogisticRegression()
clf.fit(X_train, y_train)

print('Number of dimensions before PCA: ' + str(len(pixel_colnames)))
print('Number of dimensions after PCA: ' + str(pca.n_components_))
print('Classification accuracy: ' + str(clf.score(X_test, y_test)))

#==========================================================================
# A Plot to demonstrate the cumulative explained variance and the
# number of principal components

# if n_components is not set, all components are kept (784 in this case)
pca = PCA()

pca.fit(scaledTrainImages)

# Summing explained variance
tot = sum(pca.explained_variance_)

var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)]

# Cumulative explained variance
cum_var_exp = np.cumsum(var_exp)

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7));
ax.tick_params(labelsize = 18)
ax.plot(range(1, 785), cum_var_exp, label='cumulative explained variance')
ax.set_ylabel('Cumulative Explained variance', fontsize = 16)
ax.set_xlabel('Principal components', fontsize = 16)
ax.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')
ax.axhline(y = 90, color='c', linestyle='--', label = '90% Explained Variance')
ax.axhline(y = 85, color='r', linestyle='--', label = '85% Explained Variance')
ax.legend(loc='best', markerscale = 1.0, fontsize = 12)


plt.show()
