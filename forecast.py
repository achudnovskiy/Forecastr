import sys
from Common.Config import Config

method = sys.argv[1]
Config.FORECAST_MODE = True

if method == "pg":
    import PG.Forecaster as PGForecaster
    PGForecaster.run()
else:
    from A3C.Server import Server as A3CTrainer
    A3CTrainer().main()
