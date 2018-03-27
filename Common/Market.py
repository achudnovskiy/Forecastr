import pandas
from pandas_datareader import data as dreader
from sklearn import preprocessing

from .Trader import Trader
from .Config import Config
import Common.FileManager

class Market:
    def __init__(self):
        self.trader = Trader(Config.BUDGET, Config.TRANSACTION_FEE)
        self.iterator = Config.SLICE_SIZE
        self.currentStock = pandas.DataFrame()
        self.totalReward = 0

    def loadStockData(self, ticker):
        self.currentStock = dreader.DataReader(ticker, Config.DATA_SOURCE, Config.STOCK_START_DATE, Config.STOCK_END_DATE)
        Config.SLICE_WIDTH = self.currentStock.shape[1] 

    def reset(self):
        self.trader = Trader(Config.BUDGET, Config.TRANSACTION_FEE)
        self.iterator = Config.SLICE_SIZE
        self.totalReward = 0
        if self.currentStock.empty:
            self.loadStockData(Config.TICKER)
        stockData, done = self.nextBatch()
        return self.preprocess(stockData)

    def currentBatchLength(self):
        return self.currentStock.shape[0]

    def nextBatch(self):    
        start_index = self.iterator - Config.SLICE_SIZE
        end_index = self.iterator
        batch = self.currentStock.iloc[start_index:end_index]
        self.iterator = self.iterator + 1
        done = self.iterator >= self.currentBatchLength()
        return batch, done

    def preprocess(self,stockData):
        scaledData = preprocessing.MinMaxScaler().fit_transform(stockData)
        flattenData = scaledData.flatten().T
        return flattenData

    def takeAction(self, action):
        stockData, done = self.nextBatch()
        reward = self.trader.takeAction(action, stockData.tail(1))
        self.totalReward = self.totalReward + reward
        # add current position to the observation
        observation = self.preprocess(stockData)
        return observation, reward, done, None
    
    def exportDataTradingData(self):
        return  self.currentStock, self.trader.exportHistory()