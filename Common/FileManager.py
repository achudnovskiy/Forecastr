import pickle
import csv

modelFiles = {"PG":"pgModel", "A3C":"a3cModel"}
tradeHistoryFile = "tradeHistory"
def saveModel(modelType, data):
    try:
        fileName = modelFiles[modelType]
        saveDataToFile(fileName, data)
    except IndexError:
        print("wrong model type")
        raise

def saveDataToFile(fileName, data):
    fileObject = open(fileName, 'wb')
    pickle.dump(data, fileObject)
    fileObject.close()

def restoreDataFromFile(fileName):
    fileObject = open(fileName, 'rb')
    try:
        return pickle.load(fileObject)
    except:
        print("No model found, creating new one")
        return None

def saveTrades(stock, trades):
    history = {
        "stock":stock,
        "trades":trades
    }
    saveDataToFile(tradeHistoryFile, history)
    
def restoreHistory():
    history = restoreDataFromFile(tradeHistoryFile)
    return history["stock"], history["trades"]