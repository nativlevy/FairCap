import logging
import os
import sys

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# <= WARNING: stdout
# > WARNING: stderr
logging.basicConfig(level=logging.DEBUG)


def init_logger(model_name):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    out_handler = logging.FileHandler(os.path.join(
        os.getenv('HOME'), 'output', 'stdout.log'))
    out_handler.setLevel(logging.DEBUG)
    out_handler.setFormatter(formatter)
    logger.addHandler(out_handler)
    logger.propagate = False
    return logger
