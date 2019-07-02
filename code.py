# --------------
# Code starts here
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder

#### Data 1
# Load the data

data1 = pd.read_csv(path)

# Overview of the data
data1.describe()

# Histogram showing distribution of car prices
sns.distplot(data1.price)

# Countplot of the make column
sns.countplot(data = data1,x = 'make')
plt.show()

# Jointplot showing relationship between 'horsepower' and 'price' of the car
sns.jointplot(x=data1['horsepower'],y=data1['price'],kind="reg")

# Correlation heat map
sns.heatmap(data1.corr(),cmap='YlGnBu')

# boxplot that shows the variability of each 'body-style' with respect to the 'price'
sns.boxplot(x=data1['body-style'],y=data1['price'])

#### Data 2

# Load the data
data2 = pd.read_csv(path2)


# Impute missing values with mean
data2 = data2.replace("?","NaN")
mean_imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
data2['normalized-losses'] = mean_imputer.fit_transform(data2[['normalized-losses']])
data2['horsepower'] = mean_imputer.fit_transform(data2[['horsepower']])

# Skewness of numeric features
from scipy.stats import skew
numeric_features = data2.select_dtypes(include=['number']).columns
for features in numeric_features:
    if skew(data2[features]) > 1:
        print(features)
        data2[features] = np.sqrt(data2[features])


# Label encode 
cat_features = data2.select_dtypes(include=['category','object']).columns
for cat_feature in cat_features:
    le = LabelEncoder()
    print(data2[cat_feature].head())
    data2[cat_feature] = le.fit_transform(data2[cat_feature])
    print(data2[cat_feature].head())
    
    
data2['area'] = data2['height'] * data2['width']
data2['area'].head()


# Code ends here






