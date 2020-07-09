from scareddataset.iotools import load_depthmap_xyz, save_subpix_png
from scareddataset.data_maniputlation import pts3d_to_depthmap
from scareddataset.calibrator import StereoCalibrator
import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser(description='convert ground truth from Scared format to one channel depthmaps to save disk space.')
parser.add_argument('input', help='path to scared depthmap tiff file.')
parser.add_argument('calib', help='path to calib file.')
parser.add_argument('output', help='path to store the resulting depthmap.')


if __name__ == "__main__":
    args = parser.parse_args()
    
    scared_depthmap = load_depthmap_xyz(args.input)
    
    depthmap_n = pts3d_to_depthmap(scared_depthmap)
    
    save_subpix_png(args.output, depthmap_n)

    

