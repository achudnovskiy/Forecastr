import pandas

class Trader:

    def __init__(self, budget, fee):
        self.openPrice= 0
        self.sharesCount = 0
        self.tradingHistory = []
        self.cumulativeReward = 0
        self.budget = budget
        self.fee = fee

    def takeAction(self, action, stock):
        timestamp = stock.index.date[0]
        price = stock['Close'].iloc[0]
        
        actionTaken = {'ts':timestamp, 'index':stock.index, 'ps':'hold'}
        reward = 0
        if self.sharesCount == 0:
            if action == 1:
                actionTaken = {'ts':timestamp, 'index':stock.index, 'ps':'open'}
                self.sharesCount = self.budget // price
                self.openPrice = price

        elif action == 2:
            actionTaken = {'ts':timestamp, 'index':stock.index, 'ps':'close'}
            reward = self.sharesCount * (price - self.openPrice) - self.fee
            self.sharesCount = 0
        
        self.cumulativeReward = self.cumulativeReward + reward
        self.tradingHistory.append(actionTaken)

        return reward

    def exportHistory(self):
        return self.tradingHistory



