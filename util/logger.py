from pytorch_lightning.loggers import (
    CSVLogger,
    CometLogger
)
from util.config import conf


def get_pl_loggers():
    c = conf.logger
    print(c.csv.save_dir)
    loggers = []
    if c.csv.enable:
        csvlogger = CSVLogger(c.csv.save_dir, name=c.csv.name)
        loggers.append(csvlogger)
    if c.comet.enable:
        cometlogger = CometLogger(
            api_key=c.comet.api_key,
            project_name=c.comet.project_name,
            experiment_name=c.comet.experiment_name,
            save_dir=c.comet.save_dir,
            offline=c.comet.offline
        )
        loggers.append(cometlogger)
    return loggers
