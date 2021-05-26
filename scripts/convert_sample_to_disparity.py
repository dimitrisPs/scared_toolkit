import argparse
from pathlib import Path
from scareddataset.calibrator import StereoCalibrator
from scareddataset.data_maniputlation import ptd3d_to_disparity, transform_pts, create_RT
from scareddataset.iotools import save_subpix_png, load_depthmap_xyz
import cv2

parser = argparse.ArgumentParser(
    description='converts scared sample to disparity')
parser.add_argument('left', help='path to left image.')
parser.add_argument('right', help='path to right image.')
parser.add_argument('scared_gt',
                    help='path to sample ground truth 3d image depthmap.')
parser.add_argument('calib', help='path to the sample\'s calibration file.')
parser.add_argument('rect_left', 
                    help='path to store the left rectified image.')
parser.add_argument('rect_right',
                    help='path to store the right rectified image.')
parser.add_argument('disparity', help='path to store the resulting disparity.')
parser.add_argument('full_calib',
                    help='path to store the new calibation file, containing rectification parameters.')
parser.add_argument('--rect_alpha', help='opencv stereo rectification alpha', default=0.9, type=float)


if __name__ == "__main__":
    args = parser.parse_args()

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    calibrator = StereoCalibrator()
    calib = calibrator.load(args.calib)
    rect_left, rect_right = calibrator.rectify(left, right, args.rect_alpha)

    pts3d = load_depthmap_xyz(args.scared_gt)
    size = pts3d.shape[:2]

    pts3d = pts3d.reshape(-1, 3)
    rot_pts3d = transform_pts(pts3d, create_RT(R=calib['R1']))
    disparity_img = ptd3d_to_disparity(
        rot_pts3d, calib['P1'], calib['P2'], size)


    Path(args.rect_left).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rect_right).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(args.rect_left, rect_left)
    cv2.imwrite(args.rect_right, rect_right)
    calibrator.save(args.full_calib)
    save_subpix_png(args.disparity, disparity_img)
