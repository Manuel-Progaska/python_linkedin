# Libraries
import pandas as pd
import numpy as np
from pandas_datareader import DataReader as wb
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Get market data
start = dt.datetime.now() - relativedelta(years=5)
end = dt.datetime.now()
tickers = ['AAPL', 'TSLA', 'AMZN', 'FB','MSFT']
df = wb(tickers,'yahoo',start,end)['Close']

# Estimate financials
def financials(assets):
    df = assets
    df = df.pct_change().dropna()
    ret = df.mean() * 252
    ret = ret.values
    cov = df.cov().values * 252
    return ret, cov

# Porfolio simularion
def portfolios_simulations(assets,numb,ret,cov):
    portf_ret = []
    portf_std = []
    portf_sharpe = []
    weights = []
    df1 = pd.DataFrame(columns=assets)
    for i in range(numb):
        w = np.random.rand(len(assets))
        w = w/w.sum()
        alpha1 = np.dot(w.T,ret)
        alpha2 = np.sqrt(np.dot(w.T,np.dot(cov,w)))
        alpha3 = alpha1/alpha2
        portf_ret.append(alpha1)
        portf_std.append(alpha2)
        portf_sharpe.append(alpha3)
        weights.append(w)
    df1 = pd.DataFrame(columns=assets, data=weights)
    df2 = pd.DataFrame(columns=['Ret', 'Std', 'Sharpe'],)
    df2['Ret'] = portf_ret
    df2['Std'] = portf_std
    df2['Sharpe'] = portf_sharpe
    df = pd.concat([df1,df2],axis=1)
    return df

# Scaterplot of porfolios
def plot_portfolios(port,sharpe,tickers):
    opt_port = list(np.round(sharpe.iloc[0,:len(tickers)].values,3))
    x = port['Std'].values
    y= port['Ret'].values
    x_op = sharpe.iloc[0,len(tickers)+1]
    y_op = sharpe.iloc[0,len(tickers)]
    plt.figure()
    plt.scatter(x, y,c=y/x,cmap='RdYlGn')
    plt.title('Portfolio optimization | ' + str(simul) + ' simulations')
    plt.ylabel('|Annual returns|')
    plt.xlabel(
        '|Annual volatility|'\
            + '\n'+ '---------------------------'\
                + '\n' + 'Assets:  ' + str(tickers)\
                + '\n' + 'Weights: ' + str(opt_port))
    sns.set_theme()
    plt.colorbar()
    plt.scatter(x_op, y_op,color='red')
    plt.show()
    
simul = 1000000
ret, cov  = financials(df)
port = portfolios_simulations(tickers,simul,ret,cov)
max_sharpe = port[port['Sharpe'] == port['Sharpe'].max()]
fig = plot_portfolios(port,max_sharpe,tickers)
