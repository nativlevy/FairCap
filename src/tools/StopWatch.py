import os, sys
import logging
import time
class StopWatch(object):
    def __init__(self, args):
        self.prefix = " ".join([str(arg) for arg in args])
        return
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.prefix, time.time() - self.start)