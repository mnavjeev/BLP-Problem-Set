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
from scipy.stats import norm
import multiprocessing
from joblib import Parallel, delayed
import itertools
num_cores = multiprocessing.cpu_count()
data = pd.read_csv('dataCleaned2.csv')

## Code for faster Groupby Operations
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result

##Want to get all the draws from the distribution of income
incomeDists = []
for i in range(1,20):
    temp = 'hhincome' + str(i)
    array = np.array(data[temp])
    incomeDists.append(array)
incomeDistTotal = np.unique(np.hstack(incomeDists))

## Helper Function for Cartesian Products
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)



## Integral Solver
## For computational simplicity, take an objective income distribution
incomeDistTotal2 = np.random.choice(incomeDistTotal, size = 10)

##Test 1: Parallel Integral Solver:

def integrand(v, inc, d, df, sB, sI):
    df['top'] = d*np.exp(sB*v*df['branded'] + inc*sI*df['price'])
    grouped = df[['market','top']].groupby('market')['top'].sum()
    df['bottom'] = pd.merge(grouped, df, on = 'market')['top_x']
    return df['top']/(df['bottom'] + 1)

def wrapper(list,d, df, sB, sI):
    df['sum'] = 0
    for i in list:
        df['sum'] = df['sum'] + integrand(i[0], i[1],d, df, sB, sI)
    return df['sum']


def sharesPar(d, sB, sI,df):
    vVec = np.random.normal(size = 1000)
    iVec = incomeDistTotal2
    inputs = list(cartesian_product(vVec,iVec))
    if __name__ == "__main__":
        processed = Parallel(n_jobs = 8)(delayed(integrand)(i[0],i[1], d, df, sB, sI) for i in inputs)
    for i in processed:
        df['sum'] = df['sum'] + i/len(inputs)
    return df['sum']

##Straightforward  First Attempt Integral Solver
def shares2(d, sB, sI,df):
    df['sum'] = 0
    vVec = np.random.normal(size = 11000)
    #iVec = np.random.choice(incomeDistTotal, size = 100)
    iVec = incomeDistTotal2
    total = numpy.transpose([numpy.tile(vVec, len(iVec)), numpy.repeat(iVec, len(vVec))])
    for tuple in total:
        df['sum'] = df['sum'] + integrand(df, tuple, sB, sI)
    return df['sum']/(len(vVec)*len(iVec))

## Hopefully faster integral solver:
def shares3(incomes, vVec, d, sB, sI, df):
    df['sum'] = 0
    init1 = np.transpose(np.tile(sB*df['branded'],(len(vVec)*len(incomes),1)))
    init2 = np.transpose(np.tile(sI*df['price'], (len(vVec)*len(incomes),1)))
    vVecLong = np.concatenate([vVec for i in incomes])
    incomesLong = np.repeat(incomes, len(vVec))
    top = d[:, np.newaxis]*np.exp(init1*vVecLong)*np.exp(init2*incomesLong)
    temp = pd.concat([df[['market']],pd.DataFrame(top, index = data.index, dtype = float)], axis = 1)
    sums = temp.groupby('market').sum() + 1
    sumsArray = sums.iloc[np.repeat(np.arange(len(sums)), 11)]
    df['sum'] = df['sum'] + (1/(len(vVec)*len(incomes)))*np.array(top/sumsArray).sum(axis = 1)
    return df['sum']

def shares3Par(d, sB, sI, df):
    iVec = incomeDistTotal2
    vVec = np.random.normal(size = 4800)
    split1 = np.hsplit(np.array(iVec), 5)
    split2 = np.hsplit(vVec, 10)
    interim = []
    for s1 in split1:
        for s2 in split2:
            interim.append([s1,s2])
    if __name__ == "__main__":
        totals = Parallel(n_jobs = 6)(delayed(shares3)(i[0], i[1], d, sB, sI, df) for i in interim)
    new = 0
    for p in totals:
        new =  new + p
    return new/len(interim)


## Iterate from one set of mean utilities to the next
def deltaNext2(d, sB, sI, df):
    predicted = shares3Par(d, sB, sI, df)
    interim = d*(df.share/predicted)
    return interim



#### Get X Matrices to Premultiply Everything
brandDummies = pd.get_dummies(data.brand, prefix = 'brand_')
Xmat = np.array(data[['price', 'prom']].join(brandDummies))
XmatT = Xmat.transpose()

#### Get Intrument Matrices
instrumentList  = ['cost']
for i in range(1,31):
    string = 'pricestore' + str(i)
    instrumentList.append(string)

Zmat = np.array(data[instrumentList])
ZmatT = Zmat.transpose()

##### Weighting matrix
omega = np.linalg.inv(np.dot(ZmatT, Zmat))

#### Construct the premultiplication matrix
first = np.linalg.inv(np.dot(np.dot(np.dot(np.dot(XmatT,Zmat), omega), ZmatT), Xmat))
second = np.dot(np.dot(np.dot(XmatT, Zmat), omega), ZmatT)
preMultMat = np.dot(first, second)


### ivEval should find the beta's and return the GMM criterion
def ivEval(d, df):
    dMat = np.array(d)
    beta = np.dot(preMultMat, dMat)
    xi = dMat - np.dot(Xmat, beta)
    outer = np.dot(ZmatT, xi)
    return np.abs(np.dot(np.dot(outer.transpose(), omega), outer))


#### Now Put It All Together

def runBLP(b11, b12, b21, b22, gridSize):
    critList = []
    print("Starting BLP")
    currentMin = 10000
    sImin = 0
    sBmin = 0
    deltaMin = 0.5
    sbG, sbI= np.linspace(b11, b12, int(np.sqrt(gridSize))), np.linspace(b21, b22, int(np.sqrt(gridSize)))
    sigmaGrid = cartesian_product(sbG, sbI) #Grid of sigma's to search over
    for tuple in sigmaGrid:
        print(tuple)
        sigmaB = tuple[0]
        sigmaI = tuple[1]
        data['delta'] = 1/12
        delta  = 0
        deltaNew = np.exp(data['delta'])
        counter = 0
        while np.linalg.norm(deltaNew - delta) > 1 and counter < 50:
            counter +=1
            delta = deltaNew
            deltaNew = deltaNext2(delta, sigmaB, sigmaI, data)
            print(np.linalg.norm(deltaNew - delta))
        delta = np.log(deltaNew)
        print('Iteration Done, Counter = ', counter)
        criterion = ivEval(delta, data)
        critList.append(criterion)
        print('Criterion Function = ', criterion)
        if criterion < currentMin:
            currentMin = criterion
            sBmin = sigmaB
            sImin = sigmaI
            deltaMin = delta
        if criterion < 0.1:
            print("sigmaB = ", sBmin)
            print("sigmaI = ", sImin)
            pd.DataFrame(deltaMin).to_csv("deltas.csv")
            return sBmin, sImin, deltaMin
    print("sigmaB = ", sBmin)
    print("sigmaI = ", sImin)
    return sBmin, sImin, deltaMin, critList, sigmaGrid

test1 = runBLP(0.0, 0.3, 0.0, 0.2, 16)

pd.DataFrame([test1[2], test2[2], test3[2]]).to_csv("deltas2.csv")
