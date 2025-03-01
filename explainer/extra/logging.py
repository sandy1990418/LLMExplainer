import sys
import logging
import colorlog


class LoggerHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"


def reset_logging():
    r"""
    Removes basic config of root logger
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    log_config = {
        "DEBUG": {"level": 10, "color": "purple"},
        "INFO": {"level": 20, "color": "green"},
        "WARNING": {"level": 30, "color": "yellow"},
        "ERROR": {"level": 40, "color": "red"},
    }
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(name)s: %(message)s",
        log_colors={key: conf["color"] for key, conf in log_config.items()},
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    return logger
