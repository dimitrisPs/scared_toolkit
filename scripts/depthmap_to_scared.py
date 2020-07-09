from scareddataset.iotools import load_subpix_png, save_depthmap_xyz
from scareddataset.data_maniputlation import depthmap_to_pts3d
from scareddataset.calibrator import StereoCalibrator
import numpy as np
import tifffile as tiff
import argparse
import cv2


parser = argparse.ArgumentParser(description='convert ground truth from Scared format to one channel depthmaps to save disk space.')
parser.add_argument('input', help='path to one channel depthmap.')
parser.add_argument('calib', help='path to calib file.')
parser.add_argument('output', help='path to store the resulting scared depthmap.')


if __name__ == "__main__":
    args=parser.parse_args()

    depthmap, _ = load_subpix_png(args.input)

    calib = StereoCalibrator().load(args.calib)
    
    scared = depthmap_to_pts3d(depthmap, calib['K1'], calib['D1'])
    
    save_depthmap_xyz(args.output, scared)
    
    
    
    
    
    