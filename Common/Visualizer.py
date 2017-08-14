# Meant to be run in iPython


# %matplotlib inline
import matplotlib.pyplot as plt
import Common.FileManager as FileManager

def visualizeTrades():
    stock, trades = FileManager.restoreHistory()
    plot(stock, trades, True)

def plot(stock, trades, excludeHold):
    ax = stock['Close'].plot(figsize=(20,10))
    for trd in trades:
        if excludeHold and trd['ps'] == 'hold':
            continue
        act_ind = trd['index']
        ax.annotate(trd['ps'],(stock.loc[act_ind].index, stock['Close'][act_ind].iloc[0]), textcoords='offset points',xytext=(15, 15),arrowprops=dict(arrowstyle='-|>'))
