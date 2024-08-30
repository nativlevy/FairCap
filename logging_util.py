import logging
import os
import sys
from consts import WORKER_OUTPUT_PATH
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# <= WARNING: stdout
# > WARNING: stderr
logging.basicConfig(level=logging.DEBUG)


def init_logger(model_name, is_prod=False):
    logger = logging.getLogger(model_name)
    out_handler = logging.FileHandler(os.path.join(
        WORKER_OUTPUT_PATH, 'stdout.log'))
    if is_prod:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.DEBUG)
        out_handler.setLevel(logging.DEBUG)
    out_handler.setFormatter(formatter)
    logger.addHandler(out_handler)
    logger.propagate = False
    return logger
