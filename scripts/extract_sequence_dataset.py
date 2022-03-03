#!/usr/bin/env python3

from pathlib import Path
from scaredtk.calibrator import StereoCalibrator
import scaredtk.io as sio
import scaredtk.convertions as cvt
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# this script will generate depthmaps and disparities without using the provided
# ground truth sequences and instead use only the keyframe files, rgb.mp4 and
# frame_data.tar.gz. This script facilitates data portability as the provided
# scene_points.tar.gz are generated using each keyframe's point_cloud.obj
# and frame_data.tar.gz thus the ground truth sequcnes can get generated locally.
# In addition, a refined pointcloud can be provided by the user to generate the 
# sequence. It is not recommended to use this script to extract test data because we could
# not always replicate exactly the provided ground truth files, possibly due to
# arithmetic presision. For the test sequence files use the dataset extruction
# script.


parser =argparse.ArgumentParser(description="Create keyframe dataset")
parser.add_argument('root_dir', help="root directory under which keyframe data are stored")
parser.add_argument('--out_dir', help='where to store the resulting dataset, if not set, generated files will be stored in src folders')
parser.add_argument('--recursive', '-r', help='scans for keyframe_* directories under root_dir and processes them all', action='store_true')
parser.add_argument('--depth','-de', help='generate_depthmap in the original frame of reference', action='store_true')
parser.add_argument('--undistort','-u', help='generate undistorted depthmap and left rgb in the original frame of reference', action='store_true')
parser.add_argument('--disparity','-di', help='generate rectified views and disparity maps', action='store_true')
parser.add_argument('--alpha', help='alpha parameter to use during stereo rectification, default=-1', type=float, default=-1)
parser.add_argument('--scale_factor', help='scale factor to use when storing subpixel pngs, default=128.0', type=float, default=128.0)# check if this produces a bug


def undistort_map(K, D, size_hw):

    return cv2.initUndistortRectifyMap(K,
                                       D,
                                       None,
                                       K,
                                       size_hw[::-1],
                                       cv2.CV_32FC1)



def main():
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    #recursively find all keyframe directories
    if args.recursive:
        keyframe_dirs = sorted([p for p in root_dir.rglob('**/keyframe*') if p.is_dir()]) 
    else:
        keyframe_dirs = [root_dir]
    for kf in tqdm(keyframe_dirs,desc='processed keyframes'):
        valid_list=[]
        # create output directories
        out_dir = Path(args.out_dir)/kf.parent.name/kf.name if args.out_dir is not None else kf
        out_dir.mkdir(exist_ok=True, parents=True)
        
        # load video video, gt and calib files
        stereo_calib = StereoCalibrator()
        calib = stereo_calib.load(kf/'endoscope_calibration.yaml')

        # Keyframe 5 is a single frame
        if (kf/'data'/'rgb.mp4').is_file():
            video_loader = cv2.VideoCapture(str(kf/'data'/'rgb.mp4'))
            tqdm.write('loading tarfile sequence, this will take a lot of time...')

            # this will create a dictionary with all the available frames
            gt_sequence = sio.Img3dTarLoader(kf/'data'/'scene_points.tar.gz')
            frame_count = len(gt_sequence)
        else:
            frame_count = 1

        # initialize undistortion maps to avoid calling cv2.undistort in each iteration
        if args.undistort:
            und_maps_left = undistort_map(calib['K1'],
                                          calib['D1'],
                                          (1024, 1280))# image size
        pixel_area = 1024*1280

        for frame_id in tqdm(range(frame_count), desc='processing frames', leave=False):
        
            if frame_count !=1:
                ret, frames = video_loader.read()
                left_img = frames[: 1024]
                right_img = frames[1024:]
                assert ret
                gt_img3d = gt_sequence[frame_id][:left_img.shape[0]]
            else:
                left_img = cv2.imread(str(kf/'Left_Image.png'))
                right_img = cv2.imread(str(kf/'Right_Image.png'))
                gt_img3d = sio.load_img3d(kf/'left_depth_map.tiff')
            
            
            assert left_img is not None

            if args.depth:
                depthmap = cvt.img3d_to_depthmap(gt_img3d)
                Path(out_dir/'data'/'left').mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(out_dir/'data'/'left'/f'{frame_id:06d}.png'), left_img)
                sio.save_subpix_png(out_dir/'data'/'depthmap'/f'{frame_id:06d}.png',
                                    depthmap, args.scale_factor)
            gt_ptcloud = cvt.img3d_to_ptcloud(gt_img3d)

            if args.undistort:

                left_rgb_undistored = cv2.remap(left_img,
                                                und_maps_left[0],
                                                und_maps_left[1],
                                                cv2.INTER_LINEAR)
                
                depthmap_undistorted = cvt.ptcloud_to_depthmap(gt_ptcloud,
                                                            calib['K1'],
                                                            calib['D1'],
                                                            left_img.shape[:2])
                
                
                sio.save_subpix_png(out_dir/'data'/'depthmap_undistorted'/f'{frame_id:06d}.png',
                                    depthmap_undistorted, args.scale_factor)
                Path(out_dir/'data'/'left_undistorted').mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(out_dir/'data'/'left_undistorted'/f'{frame_id:06d}.png'),
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
                depthmap_rectified = cvt.ptcloud_to_depthmap(ptcloud_rotated,
                                                            calib['P1'][:,:3],
                                                            np.zeros(5),
                                                            right_rect.shape[:2])
                
                
                Path(out_dir/'data'/'left_rectified').mkdir(exist_ok=True, parents=True)
                Path(out_dir/'data'/'right_rectified').mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(out_dir/'data'/'left_rectified'/f'{frame_id:06d}.png'),left_rect)
                cv2.imwrite(str(out_dir/'data'/'right_rectified'/f'{frame_id:06d}.png'),right_rect)
                sio.save_subpix_png(out_dir/'data'/'depthmap_rectified'/f'{frame_id:06d}.png',
                                    depthmap_rectified, args.scale_factor)
                sio.save_subpix_png(out_dir/'data'/'disparity'/f'{frame_id:06d}.png',
                                    disparity, args.scale_factor)
            #compute pixel coverage
            coverage = 1 - (np.count_nonzero(np.isnan(gt_img3d[...,-2]))/pixel_area)
            if coverage>=.1:
                valid_list.append(frame_id)
            
        stereo_calib.save(out_dir/'stereo_calib.json')
        np.savetxt(kf/"valid.csv", valid_list, fmt='%i', delimiter=",")
        if frame_count !=1:
            video_loader.release()    
            del video_loader

if __name__=='__main__':
    main()