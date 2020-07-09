import cv2
import numpy as np
import argparse
from pathlib import Path
from scareddataset.iotools import load_depthmap_xyz, export_ply

parser = argparse.ArgumentParser(description='transforms scared depthmap 3dimage to ply.')
parser.add_argument('depthmap', help='path to scared depthmap 3dimage.')
parser.add_argument('out', help='path to store the resulting .ply')


if __name__ == "__main__":
    args = parser.parse_args()
    
    depthmap = load_depthmap_xyz(args.depthmap)
    export_ply(args.out, depthmap.reshape(-1,3))
    
    