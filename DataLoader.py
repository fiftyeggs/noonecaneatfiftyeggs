import os
import ipdb
from pandas.io.data import DataReader
from pandas import *
import glob as glob


def load(symbol,startDate='19910428',forceDownload=False):
        symbol = symbol.lower()
        startDate = datetools.to_datetime(startDate)+datetools.bday-datetools.bday

        dataPath = "data_from_yhoo"

        #first check if data is in directory, or if there is even a directory
        if not os.path.exists(dataPath):
                print "data path doesn't exist. creating."
                os.makedirs(dataPath)

        #check if file is even there
        #get all stock symbols by reading csv names from data folder
        symbols = []
        os.chdir(dataPath+"/")
        for afile in glob.glob("*.df"):
                symbols.append(afile[:-3]) #slice out csv extension
        os.chdir("../") #reset dir
        
        #if it is, read data. right now, if you don't download
        if symbol in symbols and not forceDownload:
                
                df = DataFrame.load(dataPath+"/"+symbol+".df")        
                
                print "read "+symbol+" data from binary file"
                df = df[df.index>=startDate]
                return df

        else:
                #otherwise, redownload data from yahoo
                print symbol+" data not downloaded. downloading from yhoo now..."
                df= DataReader(symbol,'yahoo',startDate)
                #save locally
                df.save(dataPath+"/"+symbol+".df")

                #return data df
                return df
