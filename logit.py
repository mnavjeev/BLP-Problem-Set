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
from statsmodels.nonparametric.kernel_regression import KernelReg as kr
from matplotlib.pyplot import figure
from stargazer.stargazer import Stargazer
from statsmodels.iolib.summary2 import summary_col
import os
os.getcwd()
os.chdir("/Users/manunavjeevan/Desktop/UCLA/Second Year/Winter 2020/IO/Problem Set 1")
data = pd.read_csv('dataCleaned.csv')
data.head()
data
#Part 1: Logit
## Want to run a regression of logged share differences against
## price and promotion
y = data['shareDiff']
x = data[['price', 'prom']]
#x = sm.add_constant(x)
model1 = sm.OLS(y,x).fit()
print(model1.summary())
print(Stargazer([model1]).render_latex())
summary_col([model1]).as_latex()



## price, promotion, and a dummy for brand
brandDummies = pd.get_dummies(data['brand'], prefix = 'brand')
x = data[['price', 'prom']].join(brandDummies)
#x = sm.add_constant(x)
model2 = sm.OLS(y,x).fit()
print(model2.summary())
print(Stargazer([model2]).render_latex())
print(summary_col([model2]).as_latex())


## Price, promotion and store*brand
data['storeBrand'] = data.store + data['brand']/100
storeBrandDummies = pd.get_dummies(data['storeBrand'])
storeBrandDummies
x = data[['price', 'prom']].join(storeBrandDummies)
#x = sm.add_constant(x)
model3 = sm.OLS(y,x).fit()
print(model3.summary())
print(Stargazer([model3]).render_latex())



## Now want to instrument for price using wholesale cost as an instrument
### First Stage 1(predict price using wholesale cost)
y1 = data['price']
x1 = data['cost']
#x1 = sm.add_constant(x1)
pm1 = sm.OLS(y1,x1).fit()
priceHat = pm1.predict(x1)
data['priceHat1'] = priceHat

### First Second Stage (use predicted price in earlier models)
y = data['shareDiff']
#### Use priceHat and Promotion as product characteristics
x = data[['priceHat1', 'prom']]
iv1 = sm.OLS(y,x).fit()
print(iv1.summary())

### Use priceHat, promotion, and brand dummies as product characteristics
x = data[['priceHat1', 'prom']].join(brandDummies)
iv2 = sm.OLS(y,x).fit()
print(iv2.summary())

### Use priceHat, promotion, and storeBrand
x = data[['priceHat1', 'prom']].join(storeBrandDummies)
iv3 = sm.OLS(y,x).fit()
print(iv3.summary())

print(summary_col([iv1,iv2,iv3]).as_latex())
## Now want to instrument for price using hausman instrument
sample = data[['price','leaveOutPrice']].sample(n= 3000)
#y2 = data['price']
x2 = data['leaveOutPrice']
#pm1 = sm.OLS(y2,sm.add_constant(x2)).fit()
#priceHat = pm1.predict(sm.add_constant(x2))
#data['priceHat2'] = priceHat

y2 = data['price']
x2 = sm.add_constant(data['leaveOutPrice'])
pm2 = sm.OLS(y2,x2).fit()
priceHat = pm2.predict(x2)
data['priceHat2'] = priceHat


### Second Second Stage (use predicted price in earlier models)
y = data['shareDiff']
#### Use priceHat and Promotion as product characteristics
x = data[['priceHat2', 'prom']]
iv4 = sm.OLS(y,x).fit()
print(iv4.summary())

### Use priceHat, promotion, and brand dummies as product characteristics
x = data[['priceHat2', 'prom']].join(brandDummies)
iv5 = sm.OLS(y,x).fit()
print(iv5.summary())

### Use priceHat, promotion, and storeBrand
x = data[['priceHat2', 'prom']].join(storeBrandDummies)
iv6 = sm.OLS(y,x).fit()
print(iv6.summary())
print(summary_col([iv4,iv5,iv6]).as_latex())


##Use these results to estimate elasticities

#create weighted means
data['shareW'] = data['share']*data['count']/68711175
data['priceW'] = data['price']*data['count']/68711175

alpha = -0.5310
grouped = data.groupby('brand')['share', 'price', 'cost']
means = grouped.mean()
means2 = means
meansTable['price'] = means['price']*alpha
meansTable['ownPrice'] = meansTable['price']*(1 - meansTable['share'])
np.outer(meansTable.price , meansTable.share).shape
cross = meansTable.join(pd.DataFrame(np.outer(meansTable.price , meansTable.share),index = [1,2,3,4,5,6,7,8,9,10,11]))
cross
cross.ownPrice[11]
for i in range(0,11):
    cross[i][i+1] = cross.ownPrice[i+1]
cross

# Merger Analysis
## Want to use the results that we have from here to say how prices may change after mergers
## We have cost here. Optimal pricing is just a function of recovered elasticities and costs

for i in range(0,10):
    cross[i] = cross[i]*cross['share']/cross['price']

deriv = np.array(cross[[0,1,2,3,4,5,6,7,8,9,10]])
deriv
omega = np.array([[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1]])
np.linalg.inv(deriv)
mc =  means2.price + np.dot(np.linalg.inv(deriv), means.share)
mc


def validate(p):
    return np.linalg.norm(means.share + np.dot(np.dot(omega, deriv), p - mc))


import scipy.optimize as opt
find = opt.minimize(validate, means.price)
means2.price
find.x
