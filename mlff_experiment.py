import argparse
import signal
import sys

from nn.mlff import *



def signal_handler(*args):
        print("-> Process terminated, saving training")
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    passs