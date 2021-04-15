import cv2
from pathlib import Path
import numpy as np
from scareddataset.calibrator import StereoCalibrator
import argparse


parser = argparse.ArgumentParser(description='Rectify stereo pairs')
parser.add_argument('left', help='path to left image.')
parser.add_argument('right', help='path to right image.')
parser.add_argument('calib', help='path to calib file')
parser.add_argument('export_dir',help='dir to save rectified images.')



if __name__ == '__main__':


    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    stereo_calibrator = StereoCalibrator()
    stereo_calibrator.load(args.calib)

    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None: raise FileNotFoundError
    if right is None: raise FileNotFoundError
    
    rect_left, rect_right = stereo_calibrator.rectify(left, right, 0.9)

    left_out_path = export_dir / (left_path.stem + '_rectified.png')
    right_out_path = export_dir / (right_path.stem + '_rectified.png')
    calib_out_path = export_dir / 'complete_calib.json'

    cv2.imwrite(str(left_out_path), rect_left)
    cv2.imwrite(str(right_out_path), rect_right)
    if not calib_out_path.exists():
        stereo_calibrator.save(calib_out_path)