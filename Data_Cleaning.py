# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:27:42 2019

@author: KJee
"""

import pandas as pd 

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
thresh = len(data)*.6
data.dropna(thresh = thresh, axis = 1)
data.dropna(thresh = 21, axis = 0)

#imputing nulls fillna()
data.odometer.fillna(data.odometer.median())
data.odometer.fillna(data.odometer.mean())

#everything lower or uppercase
data.desc.head()
data.desc.head().apply(lambda x: x.lower())
data.desc.head().apply(lambda x: x.upper())

#use regex .extract
#use strip()
#use replace()
#split 

data.cylinders.dtype
data.cylinders.value_counts()
data.cylinders = data.cylinders.apply(lambda x: str(x).replace('cylinders','').strip())
data.cylinders.value_counts()

#change data type 
data.cylinders = pd.to_numeric(data.cylinders, errors = 'coerce')


#boxplot 
data.boxplot('price')
data.boxplot('odometer')

#outlier detection and normalization remove rows with > 99% / z score 
numeric = data._get_numeric_data()

# with no null values 
from scipy import stats
import numpy as np 

data_outliers = data[(data.price < data.price.quantile(.995)) & (data.price > data.price.quantile(.005))]

data_outliers.boxplot('price')

#remove duplcates, subset, keep, etc.
data.drop_duplicates()

#histogram
data_outliers.price.hist()

#types of normalization 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data.cylinders.values.reshape(-1,1))
scaler.transform(data.cylinders.values.reshape(-1,1))
