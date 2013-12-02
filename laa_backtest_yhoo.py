from pandas import *
from pandas.io.data import DataReader
import glob
import os
import pdb

from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
import pylab
import numpy as np

import copy

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import DataLoader

def rankByMomentum(hist_returns_dict): #TO DEBUG
    #first calculate momentum
    momentum_dict = {}
    for symbol in hist_returns_dict.keys():
        returns  = hist_returns_dict[symbol]
        momentum = (returns+1).cumprod()[-1]
        momentum_dict[symbol] = momentum
    ranked_tuples = sorted(momentum_dict.items(), key=lambda x: x[1],reverse=True) #we want high momentum to have low(best) ranks
    
    #pdb.set_trace()

    #then return {symbol:{rank, value}}
    ranked_dict = {}
    rank=1
    for pair in ranked_tuples:
        symbol = pair[0]
        value = pair[1]
        ranked_dict[symbol] = {}
        ranked_dict[symbol]["rank"] = rank
        ranked_dict[symbol]["value"] = value
        rank = rank + 1
    return ranked_dict

def rankByVariance(hist_returns_dict): #TO DEBUG
    #first calculate variance
    variance_dict = {}
    for symbol in hist_returns_dict.keys():
        returns  = hist_returns_dict[symbol]
        variance = np.var(returns)
        variance_dict[symbol] = variance
    ranked_tuples = sorted(variance_dict.items(),key=lambda x: x[1]) #we want low variance to have low(best) ranks
    
    #pdb.set_trace()

    #then return {symbol:{rank, value}}
    ranked_dict = {}
    rank=1
    for pair in ranked_tuples:
        symbol = pair[0]
        value = pair[1]
        ranked_dict[symbol] = {}
        ranked_dict[symbol]["rank"] = rank
        ranked_dict[symbol]["value"] = value
        rank = rank + 1
    return ranked_dict

def rankByCorrelation(hist_returns_dict): #TO DEBUG
    #first calculate correlations
    correlation_dict = {}
    for symbol in hist_returns_dict.keys():
        returns  = hist_returns_dict[symbol]
        avgcorrel = 0
        for other_symbol in hist_returns_dict.keys():
            if other_symbol != symbol:
                avgcorrel = avgcorrel + np.corrcoef(hist_returns_dict[other_symbol],hist_returns_dict[symbol])
        avgcorrel = float(avgcorrel[0][1])/float(len(hist_returns_dict.keys())-1)
        correlation_dict[symbol] = avgcorrel
    ranked_tuples = sorted(correlation_dict.items(),key=lambda x: x[1]) #we want low correl to have low(best) ranks
    
    #pdb.set_trace()

    #then return {symbol:{rank, value}}
    ranked_dict = {}
    rank=1
    for pair in ranked_tuples:
        symbol = pair[0]
        value = pair[1]
        ranked_dict[symbol] = {}
        ranked_dict[symbol]["rank"] = rank
        ranked_dict[symbol]["value"] = value
        rank = rank + 1
    return ranked_dict

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


###BACKTESTER
def backtest(in_lookback, N, w1, w2, w3):
    #BACKTEST LOGIC
    startDate = allDfs[symbols[0]].index[0]
    endDate = allDfs[symbols[0]].index[-1]
    dates = allDfs[symbols[0]].index
    dayCount = len(dates)


    portfolio_rets = []
    portfolio_dates = []

    #PORTFOLIO CONSTRUCTION LOGIC:
    #minimum variance: minimize variance based on past 60 days covariance matrix
    min_weight = 0 #no shorting
    total_weight = 1 #no leverage
    mvp_lookback = in_lookback
    top = N
    momentum_weight = w1
    variance_weight = w2
    correlation_weight = w3

    #for each day
    firstIdx = 0
    weights_dict = {}

    for idx in range(mvp_lookback+21,dayCount):

        #if dates[idx].month==5 and dates[idx].year==2003 and dates[idx].day > 28: #debug
        #   pdb.set_trace()

        #check if new month. rebalance monthly
        if (dates[idx]==dates[-1]) or (dates[idx]!=dates[-1] and dates[idx].month != dates[idx+1].month): #CASES: 1) last date, also last day of month (assuming we run backtest past midnight) THE DATES MUST BE MOST UP TO DATE PRICE DATA, and must be updated right before backtest (or else if price data is old and this executes on the 1st, it will NOT work), 2) not last day of data, today's month different from tomorrow's month. 





            if firstIdx==0: #find the idx of the previous month because firstIdx hasn't been set yet
                curIdx = idx
                while dates[curIdx].month == dates[curIdx-1].month:
                    curIdx-=1
                firstIdx = curIdx
            #CASE: we're in last month, but it hasn't ended yet. need the weights calced at beginning of month. remember to calc them on the LAST day of the month. solution: reset firstIdx to beginning of this unfinished month. THINK ABOUT edge cases
            #1st: see if the month turns before the end of data

            #BUG introduced when: last turn of month, we still want to calc portfolio returns. it's just when we're screening on the last day, we only want the most recent weights
            
            #SOLUTION
            month_turn_flag = True
            #if today is the last day in the data
            if (dates[idx]==dates[-1]):
                #if today is the last day in the month
                if (datetime.today().day==1): #assuming we scan after midnight
                    #calc weights, set TODAY as the firstIdx (would technically by idx+1, a date that doesn't exist yet)
                    firstIdx = idx+1
                    #month_turn_flag=False so we don't calc returns
                    month_turn_flag = False
                    #TODO: set weight dates as first date in month, not first trade date
                #if otherwise
                else:
                    #calc weights, setting the most recent first day of month as firstIdx
                    firstIdx = idx
                    #month_turn_flag=False so we don't calc returns
                    while dates[firstIdx].month==dates[firstIdx-1].month: #think about the edge cases: last day of month, first day of month. in my brain, it seems to work...
                        firstIdx-=1
                    month_turn_flag = False
                #firstIdx now points to the first day of the current month
         

            
            #DEBUG
            #if dates[idx].month==1 and dates[idx+1].month==2 and dates[idx].year == 2013:
            #    pdb.set_trace()



            

            #get list of all tradable etfs at the beginning of the period (e.g. it has a return with double type)
            tradable_symbols = []
            for symbol in symbols:
                try:
                    allDfs[symbol].ix[dates[firstIdx-mvp_lookback]] #firstIdx-mvplookback because we have to have data for the lookback too
                    tradable_symbols.append(symbol)
                except KeyError:
                    print symbol+" not tradable on "+str(dates[firstIdx-mvp_lookback])

            #sort symbols by momentum
            #tradable_symbols = sort_by_momentum(tradable_symbols, allDfs, dates, firstIdx, momentum_lookback) #v2 (easier to implement for investor): include last day of month, we just enter on close of first day of month

            #take only top symbols by momentum
            #tradable_symbols = tradable_symbols[0:top]


            #pdb.set_trace()
            


            #get historical returns
            hist_returns_dict = {}
            cum_returns_dict = {}
            for symbol in tradable_symbols:
                if month_turn_flag:
                    #get their returns (what about -C returns?)
                    old_returns = allDfs[symbol].ix[dates[firstIdx+1:idx]]['RET'] #firstIdx+1 to be conservative, entering the close of first day

                    #in case: convert cur_returns to doubles
                    cur_returns = Series([float(x) for x in old_returns.values],index=old_returns.index)
                    cum_returns = (cur_returns+1).cumprod() #get cumulative returns for the past month

                #get historical returns for covariance matrix
                old_returns2 = allDfs[symbol].ix[dates[firstIdx-mvp_lookback:firstIdx]]['RET']
                hist_returns = Series([float(x) if x != 'C' else 0 for x in old_returns2.values],index=old_returns2.index)

                #add to corresponding dictionaries
                hist_returns_dict[symbol] = hist_returns
                
                if month_turn_flag:
                    cum_returns_dict[symbol] = cum_returns
            
            
            #turn returns into a df
            hist_returns_df = DataFrame.from_dict(hist_returns_dict)

            ###NEW LOGIC
            #rank symbols by momentum (return dictionary of symbol:{rank,value})
            momentum_ranking = rankByMomentum(hist_returns_df)
            #rank symbols by variance
            variance_ranking = rankByVariance(hist_returns_df)
            #rank symbols by correlation
            correlation_ranking = rankByCorrelation(hist_returns_df)

            #pdb.set_trace()
            

            weights = {}
            if len(tradable_symbols)>1:
                #DETERMINE WEIGHTING
                li_dict = {}
                #add rankings for each symbol: Li
                for symbol in tradable_symbols:
                    weights[symbol] = 0
                    li_dict[symbol] = momentum_ranking[symbol]["rank"]*momentum_weight + variance_ranking[symbol]["rank"]*variance_weight + correlation_ranking[symbol]["rank"]*correlation_weight
                #select lowest (top) Li stocks
                li_sorted = sorted(li_dict.items(), key=lambda x: x[1])
                li_sorted = li_sorted[0:top]
                #make sure momentum isn't negative. if it is, set weight to 0 (cash)
                for li in li_sorted:
                    symbol = li[0]
                    if momentum_ranking[symbol]["value"]>0:
                        weights[symbol] = 1.0/float(top)
                #pdb.set_trace()
            else:
                weights = {tradable_symbols[0]:1}
            

            #sum test
            asum=0
            for key in weights.keys():
                asum+=weights[key]

            if asum>1.1:
                pdb.set_trace()

            weights_dict[dates[firstIdx-1]] = copy.deepcopy(weights)
            
            
            if month_turn_flag:
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
    portfolio_rets = DataFrame({'laa':portfolio_rets, 'cumret':portfolio_rets.cumprod()})
    weights_df = DataFrame.from_dict(weights_dict,orient='index')

    cumrets = portfolio_rets['cumret']

    performance_stats = {}
    performance_stats['cagr'] = pow(pow(cumrets[-1],(1.0/float(len(cumrets)))),252)
    performance_stats['sharpe'] = (portfolio_rets['laa']-1).mean()/(portfolio_rets['laa']-1).std()*252.0/sqrt(252)
    performance_stats['maxdd']=1-min(cumrets/cumrets.cummax())

    return performance_stats,cumrets, weights_df


###MAIN LOGIC
#get all stock symbols by reading csv names from data folder
symbols = []
#symbols = ['IVV','EEM','EFA','LQD','IYR','SHY','IEF','TIP']
symbols = ['VTI','EWJ','RWX','IEF','TLT','IAU','DBC','VGK','VNQ','VWO'] #quantopian list


allDfs = {}
#store all in memory (a dict of dfs). read from binary, if doesn't exist read from csv (and save binary)
for symbol in symbols:
    df= DataLoader.load(symbol,'20050101')
    shifted_dates = [(date+DateOffset(days=0)) for date in df.index] #shift dates to test path dependency
    df.index = shifted_dates
    allDfs[symbol] = df

#pdb.set_trace()

#calculate rets
for symbol in symbols:
        price = allDfs[symbol]['Adj Close']
        prev_price = price.shift(1)
        ret = price/prev_price-1
        ret = Series(ret,index=price.index)
        df=DataFrame({'RET':ret})
        allDfs[symbol]=df
        #pdb.set_trace()

print 'RETURN DATA RETRIEVED'

#get a ts of number of etfs in existence
generateTimeSeries = 0
if generateTimeSeries==1:
        print "generating time series of ETF count..."

        countTS = {}
        for date in allDfs['VTI'].index:
                count=0
                for symbol in allDfs.keys():
                        try:
                                allDfs[symbol].xs(date)
                                count+=1
                        except KeyError:
                                continue
                countTS[date] = count


        countSeries = Series(countTS)

w1_range = np.arange(0,1.2,0.2)
w2_range = np.arange(0,1.2,0.2)
w3_range = np.arange(0,1.2,0.2)
in_looback_range = np.arange(60,160,20)
N_range = np.arange(1,10,1)

###ROBUSTNESS TESTING

performance_stats,cumrets,weights_df = backtest(120,3,1,.25,.5)
