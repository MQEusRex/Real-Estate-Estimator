#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:40:20 2023

@author: jamescallahan
"""

# import packages

import pandas as pd # working with data
import numpy as np # working with arrays
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization


# import data
salesdata = pd.read_csv(r'/Users/jamescallahan/Desktop/Dr. Walshs Class/AlleghCoRealEstateSales.csv')
print(salesdata.head())

assessordata = pd.read_csv(r'/Users/jamescallahan/Desktop/Dr. Walshs Class/AlleghCoAssesor.csv')
print(assessordata.head())
OG=assessordata.copy(deep=True)

#Getting rid of ones with low and zero sales price (Not relevant to our problem)
assessordata = assessordata[assessordata['SALEPRICE'] >= 100000]


#Singling out the east end of pittsburgh (wards 7, 8, 11, and 14)
import numpy as np
import functools
def conjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

c_1 = assessordata['MUNIDESC'] ==	'7th Ward - PITTSBURGH'
c_2 = assessordata['MUNIDESC'] ==	'8th Ward - PITTSBURGH'
c_3 = assessordata['MUNIDESC'] ==	'14th Ward - PITTSBURGH'
#c_4 = assessordata['MUNIDESC'] ==	'10th Ward - PITTSBURGH'
c_5 = assessordata['MUNIDESC'] ==	'11th Ward - PITTSBURGH'

assessordata = assessordata[conjunction(c_1,c_2,c_3,c_5)]


#Residential homes only and no vacant lots or condos
assessordata = assessordata[assessordata['CLASSDESC'] == 'RESIDENTIAL']
assessordata = assessordata[assessordata['USEDESC'] != 'VACANT LAND']
assessordata = assessordata[assessordata['USEDESC'] != 'CONDOMINIUM']


#making a full-basement/no full basement binary indicator, same with basement garage
assessordata['BASEMENT'] = np.where(assessordata['BASEMENT'] == 5, 1, 0)
assessordata['BSMTGARAGE'] = np.where(assessordata['BSMTGARAGE'] >=1, 1, 0)

#Putting dates into the correct format
from datetime import datetime 
assessordata['SALEDATE'] = pd.to_datetime(assessordata['SALEDATE'])




#Only sale dates from the last 10 years
assessordata = assessordata[assessordata['SALEDATE'] >= '2016-04-03']


#Dropping na in square footage
assessor=assessordata.dropna(subset=['FINISHEDLIVINGAREA'])



#encoding some qualitative variables
assessor.loc[assessor['GRADE'] == 'C-', 'GRADE'] = 0
assessor.loc[assessor['GRADE'] == 'C', 'GRADE'] = 1
assessor.loc[assessor['GRADE'] == 'C+', 'GRADE'] = 2
assessor.loc[assessor['GRADE'] == 'B-', 'GRADE'] = 3
assessor.loc[assessor['GRADE'] == 'B', 'GRADE'] = 4
assessor.loc[assessor['GRADE'] == 'B+', 'GRADE'] = 5
assessor.loc[assessor['GRADE'] == 'A-', 'GRADE'] = 6
assessor.loc[assessor['GRADE'] == 'A', 'GRADE'] = 7
assessor.loc[assessor['GRADE'] == 'A+', 'GRADE'] = 8
assessor.loc[assessor['GRADE'] == 'X-', 'GRADE'] = 9
assessor.loc[assessor['GRADE'] == 'X', 'GRADE'] = 10
assessor.loc[assessor['GRADE'] == 'X+', 'GRADE'] = 11
assessor.loc[assessor['GRADE'] == 'XX-', 'GRADE'] = 12
assessor.loc[assessor['GRADE'] == 'XX', 'GRADE'] = 13
assessor.loc[assessor['GRADE'] == 'XX+', 'GRADE'] = 14

#Making sure square footage is large and land area is large (finding comparables)
assessor = assessor[assessor['FINISHEDLIVINGAREA'] >= 3500]
assessor = assessor[assessor['LOTAREA'] >= 15000]

#CREATING AN AGE COLUMN
assessor['YEARBLT']=2023-assessor['YEARBLT']
#Cutting it down to just the stuff we really want
assessor = assessor.iloc[:, 26:82]

assessor = assessor.iloc[:, [0,1,7,8,23,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,47,48,49,50,51,52,53,54,55]]

assessor=assessor.drop(['COUNTYTOTAL', 'FAIRMARKETBUILDING','FAIRMARKETTOTAL','FAIRMARKETLAND','STYLE','ROOFDESC','EXTFINISH_DESC','HEATINGCOOLING','USEDESC','SALEDATE','BASEMENTDESC','CDU','GRADEDESC','CONDITION','STYLEDESC'], axis=1)


#Replace NA w/ medians of the columns
assessor=assessor.fillna(assessor.median())


#Encoding qualitative variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
assessor['HEATINGCOOLINGDESC'] = le.fit_transform(assessor['HEATINGCOOLINGDESC'])



#Creating a pairplot to look at correlations as well as to see if we need to do any log transformations, etc. 
import seaborn as sns # data visualization
#sns.pairplot(assessor)



#Hedonic regression time
import statsmodels.api as sm
import numpy as np
from patsy import dmatrices

y, X = dmatrices('SALEPRICE ~  FINISHEDLIVINGAREA + LOTAREA + YEARBLT + BSMTGARAGE', 
                 data=assessor, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
residuals = res.resid
predicted = res.fittedvalues
observed = y
print(res.summary())




#Getting corresponding values for the King Estate:
    #9286 sqft
    #8 bed
    #2 bath
    #2 half-bath
    #4 fireplaces
    #3 stories
    #15 total rooms
    # No basement garage
    #lot area is 80,368 sq. ft.
    #year built: 1880 == CREATED AN AGE COLUMN = 143
    #Condition is excellent = 11
    
#Using the above hedonic regression we get that the house and property have a value of $3,050,410







#Now we will try a machine learning approach to solving this problem



# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric

sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size



#Some heat-map action
sb.heatmap(assessor.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()



#Plotting sales price to see if we need to do a log-transformation
sb.distplot(assessor['SALEPRICE'], color = 'r')
plt.title('Sale Price Distribution', fontsize = 16)
plt.xlabel('Sale Price', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('distplot.png')
plt.show()



#Moving the "y" column to the front of the dataset
column_to_move = assessor.pop("SALEPRICE")

# insert column with insert(location, column_name, column_value)

assessor.insert(0, "SALEPRICE", column_to_move)


#Removing some more unnecessary columns
assessor=assessor.drop(['ROOF', 'EXTERIORFINISH'], axis=1)



#Splitting the data into test and train sets
from sklearn.model_selection import train_test_split 

X = assessor.iloc[:, 1:15]
y = assessor.iloc[:, 0]


#taking 20% of the past playoff data to set aside
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




# MODELING

# 1. OLS

ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)

# 2. Ridge

ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

# 3. Lasso

lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

# 4. Bayesian

bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

# 5. ElasticNet

en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)




# 1. Explained Variance Score

print(cl('EXPLAINED VARIANCE SCORE:', attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Ridge model is {}'.format(evs(y_test, ridge_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Lasso model is {}'.format(evs(y_test, lasso_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Bayesian model is {}'.format(evs(y_test, bayesian_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of ElasticNet is {}'.format(evs(y_test, en_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')



#Elastic Net it is

# 2. R-squared

print(cl('R-SQUARED:', attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of ElasticNet is {}'.format(r2(y_test, en_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')




#Next move is to make a row of data that is the King Estate data and feed it into the classifier
column_headers = list(assessor.columns.values)
print("The Column Header :", column_headers)

# Insert row to the dataframe using DataFrame.append()
df = pd.DataFrame(X)
new_row = {'LOTAREA':80368, 'STORIES':3, 'YEARBLT':143, 'BASEMENT':1, 'GRADE':10, 'TOTALROOMS':15, 'BEDROOMS':8, 'FULLBATHS':2, 'HALFBATHS':2, 'HEATINGCOOLINGDESC':1, 'FIREPLACES':4, 'BSMTGARAGE':0, 'FINISHEDLIVINGAREA':9286}

X = df.append(new_row, ignore_index=True)
assessX=X.iloc[77,:]

# 5. ElasticNet

en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X)


#Using the elastic net method, we get an estimate of $3,242,150

#appending the estimates to the original data set to see how accurate they are overall
assessor= assessor.append(new_row, ignore_index=True)
assessor['estimates']=en_yhat

#Moving the estimates column to the front of the dataset
column_move = assessor.pop("estimates")
assessor.insert(0, "estimates", column_move)

column_move = assessor.pop("SALEPRICE")
assessor.insert(0, "SALEPRICE", column_move)








#Next move: add by-neighborhood fixed effects (basically accounting for the fluctuation in housing prices by neighborhood)


#getting rid of low and na prices
OG = OG[OG['SALEPRICE'] >= 100000]


#getting rid of outlier land size
OG = OG[OG['LOTAREA'] <= 150000]

#Making sure square footage is large and land area is large (and making sure is residential in the city of PGH/finding comparables)
OG = OG[OG['FINISHEDLIVINGAREA'] >= 5000]
OG = OG[OG['LOTAREA'] >= 20000]

#Sale date relatively recent

#Putting dates into the correct format
from datetime import datetime 
OG['SALEDATE'] = pd.to_datetime(OG['SALEDATE'])




#Only sale dates from the last 10 years
OG = OG[OG['SALEDATE'] >= '2016-04-03']


#Dropping na in square footage
OG=OG.dropna(subset=['FINISHEDLIVINGAREA'])

#PGH only
OG = OG[OG['PROPERTYCITY'] == 'PITTSBURGH']


#Residential homes only and no vacant lots or condos
OG = OG[OG['CLASSDESC'] == 'RESIDENTIAL']
OG = OG[OG['USEDESC'] != 'VACANT LAND']
OG = OG[OG['USEDESC'] != 'CONDOMINIUM']


#encoding some qualitative variables
OG.loc[OG['GRADE'] == 'D-', 'GRADE'] = -2
OG.loc[OG['GRADE'] == 'C-', 'GRADE'] = 0
OG.loc[OG['GRADE'] == 'C', 'GRADE'] = 1
OG.loc[OG['GRADE'] == 'C+', 'GRADE'] = 2
OG.loc[OG['GRADE'] == 'B-', 'GRADE'] = 3
OG.loc[OG['GRADE'] == 'B', 'GRADE'] = 4
OG.loc[OG['GRADE'] == 'B+', 'GRADE'] = 5
OG.loc[OG['GRADE'] == 'A-', 'GRADE'] = 6
OG.loc[OG['GRADE'] == 'A', 'GRADE'] = 7
OG.loc[OG['GRADE'] == 'A+', 'GRADE'] = 8
OG.loc[OG['GRADE'] == 'X-', 'GRADE'] = 9
OG.loc[OG['GRADE'] == 'X', 'GRADE'] = 10
OG.loc[OG['GRADE'] == 'X+', 'GRADE'] = 11
OG.loc[OG['GRADE'] == 'XX-', 'GRADE'] = 12
OG.loc[OG['GRADE'] == 'XX', 'GRADE'] = 13
OG.loc[OG['GRADE'] == 'XX+', 'GRADE'] = 14

#Replace NA w/ medians of the columns
OG['GRADE']=OG['GRADE'].astype(int)

OG=OG.fillna(OG.median())

#CREATING AN AGE COLUMN
OG['YEARBLT']=2023-OG['YEARBLT']

#Filtering out the mcmansion types
OG = OG[OG['YEARBLT'] >= 80]

#making a full-basement/no full basement binary indicator, same with basement garage
OG['BASEMENT'] = np.where(OG['BASEMENT'] == 5, 1, 0)
OG['BSMTGARAGE'] = np.where(OG['BSMTGARAGE'] >=1, 1, 0)

#Starting by grouping by neighborhood code (using mean)
DF1=OG.groupby(['PROPERTYZIP']).median()
DF1['PROPERTYZIP'] = DF1.index
DF1 = DF1.reset_index(drop=True)

#create a variable for Highland Park mean sale price - NEIGHCODE mean saleprice
#15206 is our zip code of interest
DF1['PRICECONTROL']=DF1.iloc[1]['SALEPRICE']-DF1['SALEPRICE']


#Apply pricecontrol variable to the OG dataset by zip code
column_move = DF1.pop("PRICECONTROL")
DF1.insert(0, "PRICECONTROL", column_move)
column_move = DF1.pop("PROPERTYZIP")
DF1.insert(0, "PROPERTYZIP", column_move)

DF1 = DF1.iloc[:, 0:2]
OG = pd.merge(OG,DF1[['PROPERTYZIP','PRICECONTROL']],on='PROPERTYZIP', how='left')

# Insert row to the dataframe using DataFrame.append()
#df = pd.DataFrame(OG)
#new_row = {'LOTAREA':80368, 'STORIES':3, 'YEARBLT':143, 'BASEMENT':1, 'GRADE':10, 'TOTALROOMS':15, 'BEDROOMS':8, 'FULLBATHS':2, 'HALFBATHS':2, 'HEATINGCOOLINGDESC':1, 'FIREPLACES':4, 'BSMTGARAGE':0, 'FINISHEDLIVINGAREA':9286,'PRICECONTROL':0}

#OG = df.append(new_row, ignore_index=True)

#Running a simple OLS

y, X = dmatrices('SALEPRICE ~ FINISHEDLIVINGAREA + PRICECONTROL', 
                 data=OG, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
residuals = res.resid
predicted = res.fittedvalues
observed = y
print(res.summary())


#This, controlling for neighborhood comparable home prices gives us a value of $2,187,320
