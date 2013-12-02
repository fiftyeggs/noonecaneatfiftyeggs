from pandas import *
import glob
import os
import pdb

from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
import pylab

import copy

def cumprod_to_returns(cumprod):
        returns = [cumprod.values[0]]
        returns.extend([cumprod.values[i]/cumprod.values[i-1] for i in range(1,len(cumprod))])
        return Series(returns,index=cumprod.index)
        
def getVolatility(symbol, allDfs, dates, endIdx, lookback):
        returns = allDfs[symbol].ix[dates[endIdx-lookback:endIdx]]['RET']
        returns = Series([float(x) if x != 'C' else 0 for x in returns.values],index=returns.index)
        stddev = returns.std()
        return stddev

def sort_by_momentum(symbols, allDfs, dates, idx, lookback):
        momentum_dict = {}

        for symbol in symbols:

                #if symbol=='SPY' and len(symbols)>1:
                #        pdb.set_trace()

                returns = allDfs[symbol].ix[dates[idx-lookback:idx]]['RET']
                returns = Series([float(x) if x != 'C' else 0 for x in returns.values],index=returns.index)
                momentum = (returns+1).cumprod()[-1]-1

                momentum_dict[symbol] = momentum

                #if symbol=='SPY' and len(symbols)>1:        
                #        pdb.set_trace()

        sorted_symbols = list(sorted(momentum_dict, key=momentum_dict.__getitem__, reverse=True)) #sort keys by values
        
        return sorted_symbols

#get all stock symbols by reading csv names from data folder
symbols = []
os.chdir("data/")
for afile in glob.glob("*.csv"):
        symbols.append(afile[:-4]) #slice out csv extension
os.chdir("../") #reset dir

allDfs = {}
#store all in memory (a dict of dfs). read from binary, if doesn't exist read from csv (and save binary)
for symbol in symbols:
        try:
                print "reading "+symbol+" data from binary file..."
                df = DataFrame.load("data/"+symbol+".df")
        except IOError:
                print "reading "+symbol+" data from CSV..."
                df = DataFrame.from_csv("data/"+symbol+".csv")
                df.save("data/"+symbol+".df")
        allDfs[symbol] = df

#get a ts of number of etfs in existence
generateTimeSeries = 0
if generateTimeSeries==1:
        print "generating time series of ETF count..."

        countTS = {}
        for date in allDfs['SPY'].index:
                count=0
                for symbol in allDfs.keys():
                        try:
                                allDfs[symbol].xs(date)
                                count+=1
                        except KeyError:
                                continue
                countTS[date] = count


        countSeries = Series(countTS)

#set start-end date
startDate = allDfs['SPY'].index[0]
endDate = allDfs['SPY'].index[-1]
dates = allDfs['SPY'].index
dayCount = len(dates)


portfolio_rets = []
portfolio_dates = []

#PORTFOLIO CONSTRUCTION LOGIC:
#minimum variance: minimize variance based on past 60 days covariance matrix
min_weight = 0 #no shorting
total_weight = 1 #no leverage
mvp_lookback = 60
top = 5
momentum_lookback = 120

#for each day
firstIdx = 0
weights_dict = {}

for idx in range(max(momentum_lookback,mvp_lookback)+21,dayCount):

        #if dates[idx].month==5 and dates[idx].year==2003 and dates[idx].day > 28: #debug
        #        pdb.set_trace()

        #check if new month. rebalance monthly
        if dates[idx]==dates[-1] or (dates[idx].month != dates[idx+1].month):
                if firstIdx==0: #find the idx of the previous month because firstIdx hasn't been set yet
                        curIdx = idx
                        while dates[curIdx].month == dates[curIdx-1].month:
                                curIdx-=1
                        firstIdx = curIdx

                #get list of all tradable etfs at the beginning of the period (e.g. it has a return with double type)
                tradable_symbols = []
                for symbol in symbols:
                        try:
                                allDfs[symbol].ix[dates[firstIdx-mvp_lookback]] #firstIdx-mvplookback because we have to have data for the lookback too
                                tradable_symbols.append(symbol)
                        except KeyError:
                                print symbol+" not tradable on "+str(dates[firstIdx-mvp_lookback])

                #sort symbols by momentum
                tradable_symbols = sort_by_momentum(tradable_symbols, allDfs, dates, firstIdx, momentum_lookback)
                #take only top symbols by momentum
                tradable_symbols = tradable_symbols[0:top]

                #pdb.set_trace()
                
                #NEW: calculate covariance matrix (pandas has a method)
                #by getting returns (historical)
                hist_returns_dict = {}
                cum_returns_dict = {}
                for symbol in tradable_symbols:
                        #get their returns (what about -C returns?)
                        old_returns = allDfs[symbol].ix[dates[firstIdx+1:idx]]['RET'] #firstIdx+1 to be conservative, entering the close of first day
                        #old_returns = allDfs[symbol].ix[dates[firstIdx:idx-1]]['RET'] #firstIdx+1 to be conservative, entering the close of first day
                        #in case: convert cur_returns to doubles
                        cur_returns = Series([float(x) for x in old_returns.values],index=old_returns.index)
                        cum_returns = (cur_returns+1).cumprod() #get cumulative returns for the past month

                        #get historical returns for covariance matrix
                        old_returns2 = allDfs[symbol].ix[dates[firstIdx-mvp_lookback:firstIdx]]['RET']
                        hist_returns = Series([float(x) if x != 'C' else 0 for x in old_returns2.values],index=old_returns2.index)

                        #add to corresponding dictionaries
                        hist_returns_dict[symbol] = hist_returns
                        cum_returns_dict[symbol] = cum_returns
                
                
                #turn returns into a df
                hist_returns_df = DataFrame.from_dict(hist_returns_dict)
                
                weights = {}
                if len(tradable_symbols)>1:
                        cov_matrix = hist_returns_df.cov()
                        weight_symbols = cov_matrix.index.values.tolist() #save the symbols
                        S = matrix([[float(x) for x in cov_matrix.ix[i].values] for i in range(0,len(cov_matrix))])

                        n = len(tradable_symbols)
                        #NEW: optimize quadratic program to get minimum variance portfolio weights
                        #pbar: column vector of returns (zero, because we are looking for minimum variance)
                        pbar = matrix(0.0, (n,1))
                        #G: negative identity matrix. h: a column vector of zeroes. ensures there are no short positions
                        G = matrix(0.0, (n,n))
                        G[::n+1] = -1.0
                        h = matrix(0.0, (n,1))
                        #A: a row vector of 1s. b: 1. ensures that weights sum to 1 (no leverage)
                        A = matrix(1.0, (1,n))
                        b = matrix(1.0)

                        pre_weights = qp(S, pbar, G, h, A, b)['x']

                        
                        
                        for i in range(0,len(weight_symbols)):
                                weights[weight_symbols[i]]=pre_weights[i]
                        #pdb.set_trace()
                else:
                        weights = {tradable_symbols[0]:1}
                

                #sum test
                asum=0
                for key in weights.keys():
                        asum+=weights[key]

                if asum>1.1:
                        pdb.set_trace()

                weights_dict[dates[firstIdx]] = copy.deepcopy(weights)
                
                

                #then calculate weighted average returns based on volatility scales
                first_symbol = tradable_symbols.pop()
                average_returns = cum_returns_dict[first_symbol]*weights[first_symbol]
                
                for symbol in tradable_symbols:
                        #take WEIGHTED average (weighted by volatility weights, determined before) of the CUM PROD (every day). this is the portfolio return
                        average_returns += cum_returns_dict[symbol]*weights[symbol]
                        
                #these are cum prod returns (e.g. levels). transform back into returns
                average_returns = cumprod_to_returns(average_returns)

                #set the beginning of next month
                firstIdx = idx+1

                #add these daily returns to a series
                portfolio_rets.extend(average_returns.values)
                portfolio_dates.extend(average_returns.index)

#convert to a df
portfolio_rets = Series(portfolio_rets,index=portfolio_dates)
portfolio_rets = DataFrame({'mommvp':portfolio_rets})
weights_df = DataFrame.from_dict(weights_dict,orient='index')
cumrets = portfolio_rets['mommvp'].cumprod()

performance_stats = {}
performance_stats['cagr'] = pow(pow(cumrets[-1],(1.0/float(len(cumrets)))),252)
performance_stats['sharpe'] = (portfolio_rets['mommvp']-1).mean()/(portfolio_rets['mommvp']-1).std()*252.0/sqrt(252)
performance_stats['maxdd']=1-min(cumrets/cumrets.cummax())
