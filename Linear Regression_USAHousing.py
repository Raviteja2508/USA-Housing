# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 06:27:40 2018

@author: lenovo
"""
##   Problem statement

###Create a model to predict housing prices based off of existing feature


########Import Libraries
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib.inline

##


########Import Libraries
USAhousing =pd.read_csv("C:\\Users\\ravi\\Desktop\\INDRAS ACADEMY\\phyton programming\\USA_Housing.csv")

USAhousing.head()
USAhousing.describe()
USAhousing.columns
pd.isnull(USAhousing)
USAhousing.isnull().values.any()

#### Let's create some simple plots to check out the data
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr(), annot=True)

USAhousing.corr()

################Training a Linear Regression Model
### begin to train out regression model! We will need to first split up 
#our data into an X array that contains the features to train on, 
#and a y array with the target variable, 
#in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

##########Train Test Split
#from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

######Creating and Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

### Model Evaluation
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.
# print the intercept
print(lm.intercept_)

 
lm.coef_
X_train.columns
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

#### Predictions from our Model
  
predictions=lm.predict(X_test)
predictions[3]
y_test
plt.scatter(y_test,predictions)

## mean square Error
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)
 

##Residual Histogram

sns.distplot((y_test-predictions),bins=50);

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('r_square_value :',r_squared)

## MAE - Mean Average error - Understand average error






