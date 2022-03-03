import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from scaredtk.calibrator import StereoCalibrator
import scaredtk.io as sio
import scaredtk.convertions as cvt
from multiprocessing import Pool
from itertools import repeat

def compute_distort_maps(src_k, dst_k, dst_d,h,w):
    # compute distortion maps. Essentailly we need to estimate the inverse transformation
    xvalues = np.arange(w)
    yvalues = np.arange(h)
    xx, yy = np.meshgrid(xvalues, yvalues)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    maps= np.squeeze(cv2.undistortPoints(np.hstack((xx,yy)).astype(np.float32), dst_k, dst_d))
    maps= maps.reshape(h,w,2)
    return (src_k[0,0]*maps[...,0])+src_k[0,2], (src_k[1,1]*maps[...,1])+src_k[1,2]


def naive_interpolation(img):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    h,w = img.shape[:2]
    img = img.reshape(-1)
    
    ok = ~np.isnan(img)
    xp = ok.ravel().nonzero()[0]
    fp = img[~np.isnan(img)]
    x  = np.isnan(img).ravel().nonzero()[0]

    img[np.isnan(img)] = np.interp(x, xp, fp)

    return img.reshape(h,w)

def disparity_to_depth_save(disparity_p, depth_p, distortion_map_x, distortion_map_y, calib, scale_factor):

        disparity = sio.load_subpix_png(disparity_p, scale_factor)
        
        pt_cloud = cvt.disparity_to_ptcloud(disparity, calib['Q'])

        # rotate ptcloud to the original frame of reference
        pt_cloud = cvt.transform_pts(pt_cloud, cvt.create_RT(R=np.linalg.inv(calib['R1'])))
        
        
        # project project the pointcloud back to the original frame of reference
        # Depending on the rectification and geometry, this projection may have hole
        img_3d = cvt.ptcloud_to_img3d(pt_cloud, calib['P1'][:3,:3], np.zeros_like(calib['D1']), disparity.shape[:2])
        # Adjust projection for initial camera matrix and distortions
        out_img = cv2.remap(img_3d[...,-1], distortion_map_x, distortion_map_y,0)
        # interpolate missing pixels if any.
        out_img = naive_interpolation(out_img)
        #save depthmap
        sio.save_subpix_png(depth_p, out_img, scale_factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('disparity_dir', help='path to the predicted disparities', type=str)
    parser.add_argument('dst_depth_dir', help='path to store the output depthmaps', type=str)
    parser.add_argument('calibration', help='path to calibration file created during the rectification process', type=str)
    parser.add_argument('--overwrite', '-o', help='overwrite existing files in dst_depth_dir', action='store_true')
    parser.add_argument('--scale_factor', help='scale factor used to save disparities as 16bit png, default=128.0', default=128, type=float)
    parser.add_argument('--jobs', '-j', type=int, help='how many jobs to run in parallel', default=16)
    args = parser.parse_args()

    if not Path(args.disparity_dir).is_dir():
        print('disparity_dir does not exist, check the input arguments')
        return 1
    if not Path(args.calibration).is_file():
        print('calibration does not found, check the input arguments')
        return 1    
    # create depth folders
    dst_dir = Path(args.dst_depth_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    calib = StereoCalibrator().load(args.calibration)

    # precompute distortion maps to export depthmap in the original frame of reference
    # including distortions.
    distortion_map_x, distortion_map_y = compute_distort_maps(calib['P1'][:3,:3],
                                                              calib['K1'],
                                                              calib['D1'],
                                                              1024,1280)

    disparity_paths = sorted([p for p in Path(args.disparity_dir).iterdir() if p.is_file()])
    depth_paths = [dst_dir/(p.stem+'.png') for p in disparity_paths]
    
    with Pool(args.jobs) as pool:
        pool.starmap(disparity_to_depth_save, tqdm(zip(disparity_paths,
                                             depth_paths,
                                             repeat(distortion_map_x),
                                             repeat(distortion_map_y),
                                             repeat(calib),
                                             repeat(args.scale_factor)), total=len(disparity_paths)))

    return 0   
        
    


if __name__ == "__main__":
    sys.exit(main())