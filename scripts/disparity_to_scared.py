import cv2
import numpy as np
import argparse
from scareddataset.calibrator import StereoCalibrator
from scareddataset.iotools import load_subpix_png, export_ply, save_depthmap_xyz
from scareddataset.data_maniputlation import create_RT, transform_pts
import tifffile as tiff


parser = argparse.ArgumentParser(
    description='convert disparities to distorted original view depthmaps')
parser.add_argument('disparity', help='disparity to convert')
parser.add_argument(
    'calib', help='complete calibration file containing rectification parameteres.')
parser.add_argument('out', help='path to store the output scared depthmap.')


if __name__ == "__main__":
    args = parser.parse_args()
    calib = StereoCalibrator().load(args.calib)
    disparity, _ = load_subpix_png(args.disparity)

    scared_depthmap = disparity_to_original_scared(disparity, calib)

    save_depthmap_xyz(args.out, scared_depthmap)
