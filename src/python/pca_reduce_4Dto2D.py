import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

# Load the Dataset

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

speciesDict = {0: 'setosa', 1:'versicolor', 2:'virginica'}

df.loc[:,'target'] = df.loc[:, 'target'].apply(lambda x: speciesDict[x])

print(df.head())
#
# Standardize the data
#
# PCA is effected by scale so you need to scale the features in the data before using PCA. 
# You can transform the data onto unit scale (mean = 0 and variance = 1) for better performance. 
# -> StandardScaler will help standardise the dataset features.
# Apply Standardization to features matrix X
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']
x = df.loc[:, features].values
y = df.loc[:,['target']].values
x = StandardScaler().fit_transform(x)
#
# PCA projection to 2D
#
# The original data has 4 columns (sepal length, sepal width, petal length, and petal width). 
# The code below projects the original data into 2 dimensions. 
# Note that after dimensionality reduction, there usually isn’t a particular meaning assigned to each principal component. 
# The new components are just the two main dimensions of variation.
pca = PCA(n_components=2)

# Fit and transform the data
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Plot it up
# PCA projection to 2D to visualize the data set
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8));
targets = df.loc[:, 'target'].unique()
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
ax.legend(targets)
ax.grid()

# Explained Variance
# this tells us how much information/variance can ber attributed to each of the principal
# components. It gives you some idea how much information/variance might be lost from 
# the reduction in dimensions
print("explained_variance_ratio:",pca.explained_variance_ratio_)
print("sum of explained variance ratio:",sum(pca.explained_variance_ratio_))
# The two principal components account for about 96% of the varaince. 
# The first component accounts for 73%
# second compenent accounts for 23%
plt.show()
