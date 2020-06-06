import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind
import seaborn as sns
from scipy import stats
from matplotlib.pyplot import figure
figure(num=None, figsize=(14, 10), dpi=500, facecolor='w', edgecolor='k')

data1 = pd.read_csv('PS1_Data/OTC_data.csv', sep='\t')
data2 = pd.read_csv('PS1_Data/OTCDataInstruments.csv', sep='\t')
data3 = pd.read_csv('PS1_Data/OTCDemographics.csv',sep='\t')
data = pd.merge(pd.merge(data1, data2, how = 'inner'), data3, how = 'inner')

data1.head()
data2.head()
data3.head()
data.head()


#Create leave out out prices and markets
#To create leave out prices create average prices then substract your own price/30
data['leaveOutAv'] = 0
for i in range(1,31):
    tempString = 'pricestore' + str(i)
    data['leaveOutAv'] = data['leaveOutAv'] + data[tempString]/30

data['leaveOutAv']

data['leaveOutAv'] = data['leaveOutAv'] - data['price_']/30

data['leaveOutAv']

data['market'] = data.store + data.week/50
data.to_csv('dataTotal.csv')


#LOGIT
## Note that outside option market share is 38%
#Create Market Shares by Brand for Each Market
