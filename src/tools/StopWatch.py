import os, sys
import logging
import time
class StopWatch(object):
    def __init__(self):
        return
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(time.time() - self.start)