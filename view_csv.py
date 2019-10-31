import numpy as np
import argparse
from matplotlib import pyplot as plt

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('fn', help = 'the file to view')
   parser.add_argument('--threshold', help = 'An optional threshold to see binary predictions. It should be a floating point value such that 0<=threshold<1', type = float, default = None)
   return(parser.parse_args())

def main():
   args = parse_args()
   inarr = np.loadtxt(args.fn)
   if args.threshold is not None:
      inarr = np.greater(inarr, args.threshold)
   plt.imshow(inarr)
   if args.threshold is None:
      plt.colorbar()
   plt.show()

if __name__ == '__main__':
   main()