import argparse
from scareddataset.iotools import load_depthmap_xyz, save_subpix_png
from scareddataset.calibrator import StereoCalibrator
from scareddataset.data_maniputlation import create_RT, transform_pts, ptd3d_to_disparity

parser = argparse.ArgumentParser(
    description='create disparity from complete calibration and xyz file.')
parser.add_argument(
    'calib', help='path to complete calibration file, containing rectification parameters.')
parser.add_argument(
    'xyz', help='path to the provided ground truth xyz depthmap file.')
parser.add_argument('out', help='path to save the resulting disparity')


if __name__ == "__main__":
    args = parser.parse_args()

    xyz = load_depthmap_xyz(args.xyz)
    size = xyz.shape[:2]

    calib = StereoCalibrator().load(args.calib)

    # rotate gt by R1 to allign it with the rectified left frame
    xyz = xyz.reshape(-1, 3)
    rot_xyz = transform_pts(xyz, create_RT(R=calib['R1']))

    disparity_img = ptd3d_to_disparity(rot_xyz, calib['P1'], calib['P2'], size)

    save_subpix_png(args.out, disparity_img)
