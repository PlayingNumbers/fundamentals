# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:17:05 2019

@author: KJee
"""

#load in the pandas module 
import pandas as pd 

# read in data from workign directory (folder in top right)
# can read in from anywhere if full path is the pd.read_csv()
data = pd.read_csv('craigslistVehicles.csv')

#view columns & rename columns 
data.columns
data.rename(index=str,columns={"url":"new_url"})
data.rename(index=str,columns={"new_url":"url"})

# view all rows for one column
data['url']

#view all columns for select group of rows
data[0:10]

#filter for multiple columns (all below do the same thing ) 
data[['url','city','price']]
data.loc[:,['url','city','price']]
data.iloc[:,0:3]

#filter by rows and columns 
data.loc[0:100,['url','city','price']]
data.iloc[0:100,0:3]

#filter by column list 
data[data.columns]


#drop / add column #inplace = True 
#axis & inplace 
data.drop('url', axis = 1)
data.drop(['url','price'], axis = 1)

#add column 
data['age'] = 2019 - data['year']

#filtering data by columns & boolean indexing 
data[(data['age'] < 5)]

data.loc[(data.age <5),:]

# basic operators on columns 
data['price_per_mile'] = data['price'] / data['odometer']

# apply function 

def timex2(x):
    return 2*x

data['price2x'] = data['price'].apply(timex2)
data['price'].head()
data['price2x'].head()

#lambda function 
data['price3x'] = data['price'].apply(lambda x: x*3)
data['price3x'].head()

#tenary operator 
data['expensive'] = data['price'].apply(lambda x: 'expensive' if x > 10000 else 'cheap')

data['newandcheap'] = data.apply(lambda x: 'yes' if x['price'] < 10000 and x['age'] < 5 else 'no', axis = 1)
data['newandcheap2'] = data[['price','age']].apply(lambda x: 'yes' if x[0] < 10000 and x[1] < 5 else 'no', axis = 1)

#binning pd.cut / pd.qcut
pd.qcut(data.price,5) #even number 
pd.cut(data.price,5 ) #even spacing 

#dummy variables 
data_dummies = pd.get_dummies(data[['price','year','fuel','transmission','type']])

#pivot table / sort_index / sort_values 
data.pivot_table(index='year',columns='type',values='price',aggfunc ='mean').sort_index(ascending=False)
data.pivot_table(index='year',columns='type',values='price',aggfunc ='count').sort_index(ascending=False)
data.pivot_table(index='year',columns='type',values='price',aggfunc ='count').sort_index(ascending=False).plot()

#groupby 
data.groupby('type').mean()
data.groupby(['type','fuel']).mean()
data.groupby(['type','fuel'],as_index = False).mean()

# pd.merge == to join in sql  
df1 = data[['url','city']]
df2 = data[['url','price']]

df_merged = pd.merge(df1,df2,on='url')


#append and concatenate (pd.concat / pd.append)
data100 = data.sample(100, random_state = 1)
data1002 = data.sample(100, random_state = 2)

data100.append(data1002)
pd.concat([data100,data1002], axis = 0)

# write to a csv file pd.to_csv()
data100.to_csv('data100.csv')

