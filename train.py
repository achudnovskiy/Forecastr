import sys
from Common.Config import Config

method = sys.argv[1]
Config.FORECAST_MODE = False

if method == "pg":
    import PG.Trainer as PGTrainer
    PGTrainer.run()
else:
    from A3C.Server import Server as A3CTrainer
    A3CTrainer().main()
