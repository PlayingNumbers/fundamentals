# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:37:26 2019

@author: Ken
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:27:42 2019

@author: KJee
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('craigslistVehicles.csv')
data.columns

data.describe()

#remove duplcates 
data.drop_duplicates(inplace= True)

#check for nulls / % of nulls 

data.isnull().any()
data.isnull().sum()/ data.shape[0]

#remove columns with certain threshold of nulls
#threshold is the number of columns or rows without nulls 
"""thresh = len(data)*.6
data.dropna(thresh = thresh, axis = 1)
data.dropna(thresh = 21, axis = 0)
"""

#everything lower or uppercase
data.desc.head()
data.desc.head().apply(lambda x: x.lower())
data.desc.head().apply(lambda x: x.upper())
data['text_len'] = data.desc.apply(lambda x: len(str(x)))
(data['text_len'].value_counts() > 1).sum()

data.cylinders.dtype
data.cylinders.value_counts()
data.cylinders = data.cylinders.apply(lambda x: str(x).replace('cylinders','').strip())
data.cylinders.value_counts()

#change data type 
data.cylinders = pd.to_numeric(data.cylinders, errors = 'coerce')
data.cylinders.value_counts()

#boxplot 
data.boxplot('price')
data.boxplot('odometer')

data.price.max()
data.odometer.max()

#outlier detection and normalization remove rows with > 99% / z score 

# with no null values 
data_outliers = data[(data.price < data.price.quantile(.995)) & (data.price > data.price.quantile(.005)) & (data.price != 0) & (data.odometer != 0)]
data_outliers = data_outliers[(data_outliers.odometer < data_outliers.odometer.quantile(.995)) & (data_outliers.odometer > data_outliers.odometer.quantile(.005))]

#histogram
data_outliers[['price','odometer','cylinders','text_len']].hist()

#types of data cleaning  
data_outliers.isnull().sum()/data_outliers.shape[0]

#imputing nulls fillna()

data_outliers.dropna(subset=['manufacturer','make','fuel','transmission', 'title_status','year'], inplace = True)
data_outliers.isnull().sum()/data_outliers.shape[0]

data_outliers.cylinders.fillna(data_outliers.cylinders.median(), inplace = True)
data_outliers.isnull().sum()/data_outliers.shape[0]

data_outliers[['condition','VIN','drive','type','paint_color']]= data_outliers[['condition','VIN','drive','type','paint_color']].fillna('n/a')
data_outliers.isnull().sum()/data_outliers.shape[0]

data_outliers.VIN = data_outliers.VIN.apply(lambda x: 'has_vin' if x != 'n/a' else 'no_vin' )

data_final = data_outliers.drop(['city_url','url','city','size','desc','lat','long','image_url'], axis = 1)
data_final['constant'] = 1
data_final['age'] = 2019 - data_final.year
data_final.isnull().any()

numeric = data_final._get_numeric_data()

import seaborn as sns

corrdata = numeric

corr = corrdata.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#simple linear regression for year
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

#set variables need to be in specific format 
X1 = data_final.odometer.values.reshape(-1,1)
y1 = data_final.price.values.reshape(-1,1)

#create train / test split for validation 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=0)
        
reg = LinearRegression().fit(X_train1, y_train1)
reg.score(X_train1, y_train1)
reg.coef_
y_hat1 = reg.predict(X_train1)

plt.scatter(X_train1,y_train1)
plt.scatter(X_train1,y_hat1)
plt.show()

y_hat_test1 = reg.predict(X_test1)
plt.scatter(X_test1, y_test1)
plt.scatter(X_test1, y_hat_test1)
plt.show()

#MSE & RMSE penalize large errors more than MAE 
mae = mean_absolute_error(y_hat_test1,y_test1)
rmse = math.sqrt(mean_squared_error(y_hat_test1,y_test1))
print('Root Mean Squared Error = ',rmse)
print('Mean Absolute Error = ',mae)

import statsmodels.api as sm

X1b = data_final[['constant','odometer']]
y1b = data_final.price.values

X_train1b, X_test1b, y_train1b, y_test1b = train_test_split(X1b, y1b, test_size=0.3, random_state=0)

reg_sm1b = sm.OLS(y_train1b, X_train1b).fit()
reg_sm1b.summary()


#multiple linear regression 
from statsmodels.stats.outliers_influence import variance_inflation_factor

X2 = data_final[['constant','age','odometer','cylinders','text_len']]
y2 = data_final.price.values

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)

reg_sm2 = sm.OLS(y_train2, X_train2).fit()
reg_sm2.summary()

pd.Series([variance_inflation_factor(X2.values,i) for i in range(X2.shape[1])],index=X2.columns)



#actual regression 
X3 = pd.get_dummies(data_final[['constant','age','odometer','text_len','cylinders','condition','fuel','VIN','type']])
y3 = data_final.price.values

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3, random_state=0)

reg_sm3 = sm.OLS(y_train3, X_train3).fit()
reg_sm3.summary()

y_hat3 = reg_sm.predict(X_test3)

rmse3 = math.sqrt(mean_squared_error(y_hat3,y_test3))

plt.scatter(y_hat3,y_test3)

#cross validation 5 fold 
from sklearn.model_selection import cross_val_score 
X4 = pd.get_dummies(data_final[['age','odometer','cylinders','condition','fuel','VIN','type']])
y4 = data_final.price.values

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.3, random_state=0)

reg4 = LinearRegression().fit(X_train4, y_train4)
reg4.score(X_train4,y_train4)

scores = cross_val_score(reg4,X4,y4, cv=5, scoring = 'neg_mean_squared_error')
np.sqrt(np.abs(scores))






