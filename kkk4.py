# Libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import DataReader as wb
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

# Get market data
start = dt.datetime.now() - relativedelta(years=5)
end = dt.datetime.now()
tickers = ['AAPL', 'TSLA', 'AMZN', 'FB', 'MSFT']
df = wb(tickers,'yahoo',start,end)['Close']
log_ret = np.log(df/df.shift(1)).dropna()

# Functions
def stats(weights):
    mean = log_ret.mean()
    cov = log_ret.cov()
    w = np.array(weights)
    ret = np.dot(mean.T,w) * 252
    std =  np.sqrt(np.dot(w.T,np.dot(cov * 252, w)))
    return np.array([ret,std,ret/std])

def max_sharpe(weights):
    return -stats(weights)[2]

# Optimization
cons = {'type':'eq','fun':lambda x: np.sum(x) -1}
bnds = tuple((0,1) for x in range(len(tickers)))
weights = len(tickers) * [1./len(tickers)]
opts = minimize(max_sharpe, weights, method='SLSQP', bounds=bnds, constraints=cons)
op_weights = list(np.round(opts.x,3))
op_ret = np.round(stats(op_weights)[0],3)
op_std = np.round(stats(op_weights)[1],3)
op_sharpe = np.round(stats(op_weights)[2],3)

result = '\nPortfolio optimization result\n'\
        + 'Assets:  ' + str(tickers) + '\n'\
        + 'Weights: ' + str(op_weights) + '\n'\

print(result)
