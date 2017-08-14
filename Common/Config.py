import datetime

class Config:

    USE_CUDA = False

    CHECKPOINT_MARK = 100
    
    EPISODE_COUNT = 10
    FORECAST_MODE = False

    SLICE_SIZE = 30
    SLICE_WIDTH = 5

    BUDGET = 1000
    TRANSACTION_FEE = 5
    ACTIONS = 3

    DATA_SOURCE = "google"

    TICKER = "AAPL"
    TICKERS = ["MSFT", "INTC", "AAPL", "ADBE", "ADSK", "AMZN","SBUX", "QCOM", "NVDA"]
    STOCK_START_DATE = datetime.datetime(2012, 8, 1)
    STOCK_END_DATE = datetime.datetime(2017, 8, 1)
    
    POSITION_LOG_FILENAME = "position_log.txt"