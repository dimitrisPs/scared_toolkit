import argparse
from scareddataset.calibrator import StereoCalibrator
from scareddataset.data_maniputlation import ptd3d_to_disparity, transform_pts, create_RT
from scareddataset.iotools import save_subpix_png, load_depthmap_xyz
import cv2
from pathlib import Path

parser = argparse.ArgumentParser(
    description='converts scared sample to disparity')
parser.add_argument('dataset_dir', help='root_dataset_dir_folder.')


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    for dataset in dataset_dir.iterdir():
        if dataset.is_file():
            continue
        for keyframe in dataset.iterdir():
        
        
            left = cv2.imread(str(keyframe / 'Left_Image.png'))
            right = cv2.imread(str(keyframe / 'Right_Image.png'))

            calibrator = StereoCalibrator()
            calib = calibrator.load(str(keyframe / 'endoscope_calibration.yaml'))
            rect_left, rect_right = calibrator.rectify(left, right)

            pts3d = load_depthmap_xyz(str(keyframe / 'left_depth_map.tiff'))
            size = pts3d.shape[:2]

            pts3d = pts3d.reshape(-1, 3)
            rot_pts3d = transform_pts(pts3d, create_RT(R=calib['R1']))
            disparity_img = ptd3d_to_disparity(
                rot_pts3d, calib['P1'], calib['P2'], size)

            cv2.imwrite(str(keyframe / 'Left_Image_rect.png'), rect_left)
            cv2.imwrite(str(keyframe / 'Right_Image_rect.png'), rect_right)
            calibrator.save(str(keyframe / 'stereo_calib.json'))
            save_subpix_png(str(keyframe / 'disparity.png'), disparity_img)
