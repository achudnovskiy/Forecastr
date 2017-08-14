from Common.Market import Market
import random
from pprint import pprint

def testMarket():
    market = Market()
    cummulativeReward = 0
    testLength = 100
    market.reset()
    for i in range(1,testLength):
        stockData = market.nextBatch()
        action = random.randint(0,2)
        market.takeAction(action)

    # pprint(market.exportHistory(False))

testMarket()
