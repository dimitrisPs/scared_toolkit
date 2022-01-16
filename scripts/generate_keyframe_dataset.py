from pathlib import Path
from scaredtk.calibrator import StereoCalibrator, undistort
import scaredtk.io as sio
import scaredtk.convertions as cvt
import cv2
import numpy as np
import argparse
from tqdm import tqdm

parser =argparse.ArgumentParser(description="Create keyframe dataset")
parser.add_argument('root_dir', help="root directory under which keyframe data are stored")
parser.add_argument('--out_dir', help='where to store the resulting dataset, if not set, generated files will be stored in src folders')
parser.add_argument('--recursive', '-r', help='scans for keyframe_* directories under root_dir and processes them all', action='store_true')
parser.add_argument('--depth','-de', help='generate_depthmap in the original frame of reference', action='store_true')
parser.add_argument('--undistort','-u', help='generate undistorted depthmap and left rgb in the original frame of reference', action='store_true')
parser.add_argument('--disparity','-di', help='generate rectified views and disparity maps', action='store_true')
parser.add_argument('--ptcloud', help='name of the pointcloud to provide reference, .ply are supported, must be placed inside keyframe dirs.')
parser.add_argument('--alpha', help='alpha parameter to use during stereo rectification. A value of 0 results to no black borders. default=0.', type=float, default=-1)
parser.add_argument('--scale_factor', help='scale factor to use when storing subpixel pngs, default=128.0', type=float, default=128.0)



if __name__=='__main__':
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    #recursively find all keyframe dirs
    if args.recursive:
        keyframe_dirs = [p for p in root_dir.rglob('**/keyframe_*') if p.is_dir()] 
    else:
        keyframe_dirs = [root_dir]
        
    for kf in tqdm(keyframe_dirs,desc='keyframes processed'):
        out_dir = Path(args.out_dir)/kf.parent.name/kf.name if args.out_dir is not None else kf
        out_dir.mkdir(exist_ok=True, parents=True)
        
        stereo_calib = StereoCalibrator()
        calib = stereo_calib.load(kf/'endoscope_calibration.yaml')
        
        # Point clouds were construced from 3d images and those constructed by
        # triangulating corresponding pixels between stereo frames. Since
        # triangulation alters the 3D location of points, if we try to reproject
        # the provided point clouds back to images, pixels do not end up in the
        # same locations, resulting in different 3D Image. Thus for custom
        # pointcloud we need to create the 3D images, whereas for the provided
        # sequence we need to load them to maintain good pixel coverage.
        if args.ptcloud is not None:
            gt_ptcloud = sio.load_ply_as_ptcloud(kf/args.ptcloud)
            gt_img3d = cvt.ptcloud_to_img3d(gt_ptcloud,
                                            calib['K1'],
                                            calib['D1'],
                                            (1024,1280))
        else:
            gt_ptcloud = sio.load_scared_obj(kf/'point_cloud.obj')
            gt_img3d = sio.load_img3d(kf/'left_depth_map.tiff')
            
        if args.depth:
            depthmap = cvt.img3d_to_depthmap(gt_img3d)
            left_img= cv2.imread(str(kf/'Left_Image.png'))
            cv2.imwrite(str(out_dir/'Left_Image.png'), left_img)
            sio.save_subpix_png(out_dir/'depthmap.png',
                                depthmap, args.scale_factor)
                
        if args.undistort:
            left_rgb = cv2.imread(str(kf/'Left_Image.png'))
            left_rgb_undistored, _ = undistort(left_rgb,
                                               calib['K1'], calib['D1'])
            depthmap_undistorted = cvt.ptcloud_to_depthmap(gt_ptcloud,
                                                           calib['K1'],
                                                           calib['D1'],
                                                           left_rgb.shape[:2])
            
            sio.save_subpix_png(out_dir/'depthmap_undistorted.png',
                                depthmap_undistorted, args.scale_factor)
            cv2.imwrite(str(out_dir/'left_undistorted.png'), left_rgb_undistored)
            
        if args.disparity:
            left_rgb = cv2.imread(str(kf/'Left_Image.png'))
            right_rgb = cv2.imread(str(kf/'Right_Image.png'))
            left_rect, right_rect = stereo_calib.rectify(left_rgb, right_rgb,
                                                         args.alpha)
            # We need to rotate point by R1 in order to express them in the
            # left rectified frame of reference.
            ptcloud_rotated = cvt.transform_pts(gt_ptcloud,
                                                cvt.create_RT(R=calib['R1']))
            disparity = cvt.ptcloud_to_disparity(ptcloud_rotated,
                                                 calib['P1'], calib['P2'],
                                                 left_rgb.shape[:2])
            depthmap_rectified = cvt.ptcloud_to_depthmap(ptcloud_rotated,
                                                         calib['P1'][:,:3],
                                                         np.zeros(5),
                                                         left_rgb.shape[:2])
            
            cv2.imwrite(str(out_dir/'left_rectified.png'),left_rect)
            cv2.imwrite(str(out_dir/'right_rectified.png'),right_rect)
            sio.save_subpix_png(out_dir/'depthmap_rectified.png',
                                depthmap_rectified, args.scale_factor)
            sio.save_subpix_png(out_dir/'disparity.png',
                                disparity, args.scale_factor)
            stereo_calib.save(out_dir/'stereo_calib.json')    
            
        