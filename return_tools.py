import pdb as pdb
from pandas import *

def _toRets(cumrets):
        dates = []
        values = []
        for i in range(1,len(cumrets)):
                ret = cumrets.ix[i]/cumrets.ix[i-1]
                dates.append(cumrets.index[i])
                values.append(ret)
        ser = Series(values,index=dates)
        return ser

def preProcessReturns(inputRets, mode="cumrets"):
        if mode=="cumrets":
                cumrets = inputRets
                rets = _toRets(cumrets)
        else:
                rets = inputRets
                cumrets = rets.cumprod()
        
        retsDf = DataFrame({'rets':rets,'cumrets':cumrets})
        return retsDf


def generatePeriodicReturns(inputRets, mode="monthly"):
        retsDf = preProcessReturns(inputRets)
        if mode=='monthly':
                period = Series([date.month for date in retsDf.index],index=retsDf.index)
        else:
                period = Series([date.year for date in retsDf.index],index=retsDf.index)
        year = Series([date.year for date in retsDf.index],index=retsDf.index)
        retsDf['period']=period
        retsDf['year']=year
        periodRetsDict = {}

        dates = []
        values = []

        for curyear, yearslice in retsDf.groupby(year):
                for curperiod, perslice in yearslice.groupby(period):
                        
                        cumret = perslice['cumrets'][-1]/perslice['cumrets'][0]
                        dates.append(float(str(curyear)+"."+str(curperiod)))
                        values.append(cumret)

        periodicReturnsDf = Series(values,index=dates)
        return periodicReturnsDf
