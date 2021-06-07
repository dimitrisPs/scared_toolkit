from pathlib import Path
from scareddataset.calibrator import StereoCalibrator, undistort
import scareddataset.io as sio
import scareddataset.convertions as cvt
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import gc


# this script will generate depthmaps and disparities without using the provided
# ground truth sequences using only the keyframe files, rgb.mp4 and
# frame_data.tar.gz. This script facilitates data portability as the provided
# scene_points.tar.gz are generated using each keyframe's point_cloud.obj
# and frame_data.tar.gz thus can get generated locally from those. In addition
# a refined pointcloud can be provided by the user to generate the sequence.
# it is not recommended to use this script to extract test data because we could
# not always replicate exactly the provided ground truth files, possibly due to
# arithmetic presision. For the test sequence files use the dataset extruction
# script.

parser =argparse.ArgumentParser(description="Create keyframe dataset")
parser.add_argument('root_dir', help="root directory under which keyframe data are stored")
parser.add_argument('--recursive', '-r', help='scans for keyframe_* directories under root_dir and processes them all', action='store_true')
parser.add_argument('--out_dir', help='where to store the resulting dataset, if not set, generated files will be stored in src folders')
parser.add_argument('--depth','-de', help='generate_depthmap in the original frame of reference', action='store_true')
parser.add_argument('--undistort','-u', help='generate undistorted depthmap and left rgb in the original frame of reference', action='store_true')
parser.add_argument('--disparity','-di', help='generate rectified views and disparity maps', action='store_true')
parser.add_argument('--alpha', help='alpha parameter to use during stereo rectification, default=-1', type=float, default=-1)
parser.add_argument('--scale_factor', help='scale factor to use when storing subpixel pngs, default=256.0', type=float, default=256.0)
parser.add_argument('--coverage_threshold', help='drop frames with coverage less than amount [0-1]', default=0, type=float)


if __name__=='__main__':
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    #recursively find all keyframe dirs
    if args.recursive:
        keyframe_dirs = [p for p in root_dir.rglob('**/keyframe_*') if p.is_dir()] 
    else:
        keyframe_dirs = [root_dir]
        
    for kf in tqdm(keyframe_dirs,desc='processed keyframes'):
        out_dir = Path(args.out_dir)/kf.parent.name/kf.name if args.out_dir is not None else kf
        out_dir.mkdir(exist_ok=True, parents=True)
        
        stereo_calib = StereoCalibrator()
        calib = stereo_calib.load(kf/'endoscope_calibration.yaml')
        
        pose_dict = sio.load_pose_sequence(kf/'data'/'frame_data.tar.gz')
        video_loader = sio.StereoVideoCapture(kf/'data'/'rgb.mp4')
        
        tqdm.write('loading tarfile sequence, this will take a lot of time...')
        gt_sequence = sio.Img3dTarLoader(kf/'data'/'scene_points.tar.gz')
        
        
        sample_sequence = [(fid, pose_dict[fid]) for fid in pose_dict.keys()]
        
        pixel_num = np.product(sample_sequence[0][1].shape[:2])
        
        for fid, pose in tqdm(sample_sequence, desc='processing frames', leave=False):
            
            
            left_img, right_img = video_loader.read()
            
            gt_img3d = gt_sequence[fid][:left_img.shape[0]]
            
            assert left_img is not None

            if args.depth:
                depthmap = cvt.img3d_to_depthmap(gt_img3d)
                coverage = np.count_nonzero(~np.isnan(depthmap))/pixel_num
                if (args.coverage_threshold==0) or ((np.count_nonzero(~np.isnan(depthmap))/pixel_num) >= args.coverage_threshold):   
                    Path(out_dir/'left').mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_dir/'left'/f'{fid:06d}.png'), left_img)
                    sio.save_subpix_png(out_dir/'depthmap'/f'{fid:06d}.png',
                                        depthmap, args.scale_factor)
            gt_ptcloud = cvt.img3d_to_ptcloud(gt_img3d)        
            if args.undistort:
                left_rgb_undistored, _ = undistort(left_img,
                                                calib['K1'], calib['D1'])
                
                depthmap_undistorted = cvt.ptcloud_to_depthmap(gt_ptcloud,
                                                               calib['K1'],
                                                               calib['D1'],
                                                               left_img.shape[:2])
                
                
                if (args.coverage_threshold==0) or ((np.count_nonzero(~np.isnan(depthmap_undistorted))/pixel_num) >= args.coverage_threshold):  
                    sio.save_subpix_png(out_dir/'depthmap_undistorted'/f'{fid:06d}.png',
                                        depthmap_undistorted, args.scale_factor)
                    Path(out_dir/'left_undistorted').mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_dir/'left_undistorted'/f'{fid:06d}.png'),
                                    left_rgb_undistored)
                
            if args.disparity:
                left_rect, right_rect = stereo_calib.rectify(left_img, right_img,
                                                            args.alpha)
                # We need to rotate point by R1 in order to express them in the
                # left rectified frame of reference.
                ptcloud_rotated = cvt.transform_pts(gt_ptcloud,
                                                    cvt.create_RT(R=calib['R1']))
                disparity = cvt.ptcloud_to_disparity(ptcloud_rotated,
                                                    calib['P1'], calib['P2'],
                                                    right_rect.shape[:2])
                if (args.coverage_threshold==0) or ((np.count_nonzero(~np.isnan(disparity))/pixel_num)> args.coverage_threshold):   
                    depthmap_rectified = cvt.ptcloud_to_depthmap(ptcloud_rotated,
                                                                calib['P1'][:,:3],
                                                                np.zeros(5),
                                                                right_rect.shape[:2])
                    
                    
                    Path(out_dir/'left_rectified').mkdir(exist_ok=True, parents=True)
                    Path(out_dir/'right_rectified').mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_dir/'left_rectified'/f'{fid:06d}.png'),left_rect)
                    cv2.imwrite(str(out_dir/'right_rectified'/f'{fid:06d}.png'),right_rect)
                    sio.save_subpix_png(out_dir/'depthmap_rectified'/f'{fid:06d}.png',
                                        depthmap_rectified, args.scale_factor)
                    sio.save_subpix_png(out_dir/'disparity'/f'{fid:06d}.png',
                                        disparity, args.scale_factor)
                stereo_calib.save(out_dir/'stereo_calib.json')    
        # del gt_sequence
        # print(gc.collect())
        # print(gc.garbage)
            