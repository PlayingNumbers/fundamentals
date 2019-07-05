# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:46:40 2019

@author: Ken
"""



""" What is Pandas?

Pandas is a very flexible data science library that allows us to load in data, manipulate it, and analyze it. 

- Pandas uses dataframes which we can think of as tables. We can use functions to create or manipulate rows and columns. 
- Using pandas dataframes, we can begin to understand trends in data we can perform functions on the rows or columns to understand more about our set
- Pandas also has some visualization tools that we can use to better understand our data

"""

import pandas as pd 

# read in data from workign directory (folder in top right)
# can read in from anywhere if full path is the pd.read_csv()
data = pd.read_csv('craigslistVehicles.csv')

#rows and columns returns (rows, columns)
data.shape

#returns the first x number of rows when head(num). Without a number it returns 5
data.head()

#returns the last x number of rows when tail(num). Without a number it returns 5
data.tail()

#returns an object with all of the column headers 
data.columns

#basic information on all columns 
data.info()

#gives basic statistics on numeric columns
data.describe()

#shows what type the data was read in as (float, int, string, bool, etc.)
data.dtypes

#shows which values are null
data.isnull()

#shows which columns have null values
data.isnull().any()

#shows for each column the percentage of null values 
data.isnull().sum() / data.shape[0]

# for categorical variables 

#shows unique values that appear in the column 
#data.type = data['type']
data.type.unique()

#shows the counts for those unique values 
data.type.value_counts()

#shows the percentage of values from 
data.type.value_counts()/ data.type.notnull().sum()

data.cylinders.head(10)

############################################################# Graphing #######################################################
#histogram of year 
# data.year.hist() == data.year.plot(kind='hist')
data.year.hist()
data.year.hist(bins=100)

#bar chart of types 
data.type.value_counts().plot(kind='bar')

