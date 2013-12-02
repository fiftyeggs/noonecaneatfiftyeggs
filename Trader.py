

import sys
from time import sleep

from swigibpy import EWrapper, EPosixClientSocket, Contract, Order, TagValue,\
                TagValueList

import urllib as u
import string




class Trader:

        try:
                input = raw_input
        except:
                pass

        ###

        orderId = None
        availableFunds = 0
        netLiquidationValue = 0

        QUOTE_LAST = 3

        class PlaceOrderExample(EWrapper):
                '''Callback object passed to TWS, these functions will be called directly
                by TWS.

                '''

                def openOrderEnd(self):
                        '''Not relevant for our example'''
                        pass

                def execDetails(self, id, contract, execution):
                        '''Not relevant for our example'''
                        pass

                def managedAccounts(self, openOrderEnd):
                        '''Not relevant for our example'''
                        pass

                ###############

                def nextValidId(self, validOrderId):
                        '''Capture the next order id'''
                        #global orderId
                        Trader.orderId = validOrderId

                def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                                parentId, lastFilledPrice, clientId, whyHeld):

                        print(("Order #%s - %s (filled %d, remaining %d, avgFillPrice %f,"
                                   "last fill price %f)") % (
                                        id, status, filled, remaining, avgFillPrice, lastFilledPrice))

                def openOrder(self, orderID, contract, order, orderState):

                        print("Order opened for %s" % contract.symbol)

                ####account value
                def updateAccountValue(self, key, value, currency, accountName):

                        #print 'key %s' % (key)
                        #print 'value: %s' % (value)

                        #get how much current available funds we have, also our net liquidation value
                        
                        if currency == 'USD':

                                if key == 'AvailableFunds':
                                        Trader.availableFunds = float(value)
                                elif key=='NetLiquidation':
                                        Trader.netLiquidationValue = float(value)
                def accountDownloadEnd(self,accountName):
                        print 'account download ended for %s' % (accountName)
                        






        def __init__(self,accountNumber):
                self.accountNumber = accountNumber

        @staticmethod
        def get_quote(symbols):
            data = []
            url = 'http://finance.yahoo.com/d/quotes.csv?s='
            for s in symbols:
                url += s+"+"
            url = url[0:-1]
            url += "&f=sb3b2l1l"
            f = u.urlopen(url,proxies = {})
            rows = f.readlines()
            for r in rows:
                values = [x for x in r.split(',')]
                symbol = values[0][1:-1]
                bid = string.atof(values[1])
                ask = string.atof(values[2])
                last = string.atof(values[3])
                data.append([symbol,bid,ask,last,values[4]])
            return data


        def enterPositions(self,weights,execution_sleep=True):

                print "----------------------MAKING TRADES ON IB---------------------------"
                # Instantiate our callback object
                callback = self.PlaceOrderExample()

                # Instantiate a socket object, allowing us to call TWS directly. Pass our
                # callback object so TWS can respond.
                tws = EPosixClientSocket(callback)

                # Connect to tws running on localhost
                tws.eConnect("", 7496, 42)



                #account updates
                tws.reqAccountUpdates(True,self.accountNumber)

                sleep(1)
                print 'available funds: %s' % (self.availableFunds)
                print 'net liquidation value: %s' % (self.netLiquidationValue)


                
                ###DELAY UNTIL MARKET HOURS
                if execution_sleep:
                        day_of_week = datetime.now().isoweekday()

                        #if weekday, and we scanned after midnight, set execution time to this morning at 10:30 am
                        time_now = datetime.now()
                        if day_of_week in range(1,6) and (time_now.hour >= 0 and time_now.hour<10) and (time_now.minute>=0 and time_now.minute<30):
                                execution_time = datetime(year=time_now.year,month=time_now.month,day=time_now.day,hour=10,minute=30)


                        #otherwise, set to next trading day, morning at 10:30am
                        else:
                                execution_time = datetime.now()
                                execution_time = execution_time+dt.timedelta(days=1)
                                while execution_time.isoweekday()>5:
                                        execution_time = execution_time+dt.timedelta(days=1)
                                execution_time = datetime(year=execution_time.year,month=execution_time.month,day=execution_time.day,hour=10,minute=30)    


                        to_sleep = (execution_time-datetime.now()).total_seconds()
                        print "----------sleeping until execution time of %s---------------" % (execution_time)

                        #sleep until that time
                        sleep(to_sleep)








                for stock in weights:

                        print("\n=====================================================================")
                        print(" Trading "+stock)
                        print("=====================================================================\n")

                        stock_price = Trader.get_quote([stock])[0][self.QUOTE_LAST]
                        print "%s last stock price: %s" % (stock, stock_price)
                        
                        contract = Contract()
                        contract.symbol = stock
                        contract.secType = "STK"
                        contract.exchange = "SMART"
                        contract.currency = "USD"

                        if self.orderId is None:
                                print('Waiting for valid order id')
                                sleep(1)
                                while self.orderId is None:
                                        print('Still waiting for valid order id...')
                                        sleep(1)

                        # Order details

                        order = Order()
                        order.action = 'BUY'
                        #order.lmtPrice = 140
                        order.orderType = 'MKT'
                        
                        dollar_value = self.availableFunds*weights[stock]
                        order.totalQuantity = int(round(dollar_value/stock_price,0))
                        #order.algoStrategy = "AD"
                        order.tif = 'DAY'
                        #order.algoParams = algoParams
                        order.transmit = True


                        print("Placing order for %d %s's, dollar value $%s (id: %d)" % (order.totalQuantity, contract.symbol, dollar_value, self.orderId))

                        # Place the order
                        tws.placeOrder(
                                        self.orderId,                                    # orderId,
                                        contract,                                   # contract,
                                        order                                       # order
                                )
                        
                        print("\n=====================================================================")
                        print("                   Order placed, waiting for TWS responses")
                        print("=====================================================================\n")
                        
                        sleep(3)
                        #reset orderid for next
                        self.orderId=self.orderId+1

                        print("\n=====================================================================")
                        print(" Trade done.")
                        print("=====================================================================\n")


                print("******************* Press ENTER to quit when done *******************\n")
                input()

                print("\nDisconnecting...")

                tws.eDisconnect()
