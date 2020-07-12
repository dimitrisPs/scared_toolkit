import cv2
import numpy as np
import argparse
from scareddataset.iotools import load_subpix_png
from scareddataset.evaluation import depthmap_error


parser = argparse.ArgumentParser(description='evaluate scared depthmaps')
parser.add_argument('reference', help='path to ground truth depthmap')
parser.add_argument('compare', help='path to the depthmap to compare.')
parser.add_argument('out')

if __name__ == "__main__":
    args = parser.parse_args()
    
    ref = load_subpix_png(args.reference)
    comp = load_subpix_png(args.compare)
    
    avg_error = depthmap_error(ref, comp)
    print(avg_error)