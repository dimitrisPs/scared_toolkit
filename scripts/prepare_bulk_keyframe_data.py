import argparse
import cv2
import tifffile as tiff
import tarfile
from pathlib import Path
from scareddataset.calibrator import StereoCalibrator
from scareddataset.iotools import save_subpix_png, load_depthmap_xyz
from scareddataset.data_maniputlation import ptd3d_to_disparity, create_RT, transform_pts, scared_to_depthmap

parser = argparse.ArgumentParser(description='unpacks data folders and split data.')
parser.add_argument('data_folder', help='path to data folder containing video and gt tar file.')

if __name__ == "__main__":
    args= parser.parse_args()


    root_dir = Path(args.data_folder).resolve()
    gt_dir = root_dir / 'ground_truth'
    left_dir = root_dir / 'left'
    left_rect_dir = root_dir / 'left_rect'
    right_dir = root_dir / 'right'
    right_rect_dir = root_dir / 'right_rect'
    disparity_dir = root_dir/'disparity'
    depthmap_dir = root_dir/'depthmap'
    
    gt_dir.mkdir(exist_ok=True)
    left_dir.mkdir(exist_ok=True)
    left_rect_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    right_rect_dir.mkdir(exist_ok=True)
    disparity_dir.mkdir(exist_ok=True)
    depthmap_dir.mkdir(exist_ok=True)
    
    # extract gt files and and keep only left gt

        
    tar = tarfile.open(str(root_dir/'scene_points.tar.gz'), "r:gz")
    tar.extractall(str(gt_dir))
    tar.close()
    
    for gt_file in gt_dir.iterdir():
        gt = tiff.imread(str(gt_file))
        tiff.imsave(str(gt_file), gt[:1024,:,:])
    #split video frames and rectify
    
    calibrator = StereoCalibrator()
    calibrator.load(str(root_dir.parent /'endoscope_calibration.yaml'))
    calib = calibrator.calib
    
    video = cv2.VideoCapture(str(root_dir / 'rgb.mp4'))
    
    i=0
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        left = frame[:1024]
        right= frame[1024:]
        left_rect, right_rect = calibrator.rectify(left, right, 0.9)
        
        cv2.imwrite(str(left_dir / ('{:06d}.png'.format(i))), left)
        cv2.imwrite(str(right_dir / ('{:06d}.png'.format(i))), right)
        cv2.imwrite(str(left_rect_dir / ('{:06d}.png'.format(i))), left_rect)
        cv2.imwrite(str(right_rect_dir / ('{:06d}.png'.format(i))), right_rect)
        

        pts3d = load_depthmap_xyz(str(gt_dir/('scene_points{:06d}.tiff'.format(i)) ))
        depthmap_img = scared_to_depthmap(pts3d)
        size = pts3d.shape[:2]
        # rotate gt by R1 to allign it with the rectified left frame
        pts3d = pts3d.reshape(-1, 3)
        rot_pts3d = transform_pts(pts3d, create_RT(R=calib['R1']))
        disparity_img = ptd3d_to_disparity(rot_pts3d, calib['P1'], calib['P2'], size)
        
        save_subpix_png(str(disparity_dir / ('{:06d}.png'.format(i))), disparity_img)
        save_subpix_png(str(depthmap_dir / ('{:06d}.png'.format(i))), depthmap_img)
        
        i+=1
        print(i)
    video.release()
    calibrator.save(str(root_dir/'stereo_calib.json'))