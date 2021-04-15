import cv2
import argparse
import numpy as np 
import sys;sys.path.append('..');
from scareddataset import iotools
import scareddataset.data_maniputlation as dm
from pathlib import Path
from scareddataset.calibrator import StereoCalibrator
from skimage.measure import compare_ssim as ssim
import time




def specular_mask(img, lightness_val=200):
    mask =np.zeros(img.shape[:2], dtype=np.uint8)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask[img_hsl[:,:,1]<lightness_val]=255
    return mask

def left_right_consistancy_check(ref, comp):
    ref_specilar_mask = specular_mask(ref)
    comp_specilar_mask = specular_mask(comp)
    comp_ignore = np.zeros_like(ref_specilar_mask)+255
    comp_ignore[(comp==0).all(axis=2)] = 0
    
    mask = cv2.bitwise_and(ref_specilar_mask,comp_specilar_mask)
    mask = cv2.bitwise_and(mask,comp_ignore)
    ref=ref.astype(np.float32)
    comp=comp.astype(np.float32)
    ref[mask==0] =0
    comp[mask==0] = 0
    s = ssim(ref, comp, win_size=9, multichannel=True)
    ref[mask==0] = np.nan
    comp[mask==0] = np.nan
    ref = ref.reshape(-1,3)
    comp=comp.reshape(-1,3)
    result = np.sqrt(np.sum((ref-comp)**2,axis=1))
    return np.nanmean(result), s
    

def warp_left_to_right_disparity(left_in, disparity):
    start = time.time()
    right_out = np.zeros_like(left_in)
    h,w,c = left_in.shape
    for row in range(h):
        for col in range(w):
            disp = disparity[row,col]
            if (disp==0) or (col-disp<0):
                continue
            right_out[row, col-int(disp)] = left_in[row, col]
    # print(time.time()-start)
    return right_out



def warp_left_to_right_disparity_vec(left_in, disparity):
    start = time.time()
    right_out = np.zeros_like(left_in)
    h,w,c = left_in.shape
    disparity = disparity.reshape(-1)
    pixel_loc = np.mgrid[0:w,0:h].transpose(2,1,0).astype(np.float32)
    pixel_loc=pixel_loc.reshape(-1,2)#[:,::-1]
    pixel_loc2 = pixel_loc.copy()
    pixel_loc2[:,0] = pixel_loc2[:,0]- disparity
    pixel_loc2 = pixel_loc2[disparity>0]
    pixel_loc = pixel_loc[disparity>0]
    pixel_loc = pixel_loc[np.all(pixel_loc2>0, axis=1)]
    pixel_loc = pixel_loc.astype(np.int)
    pixel_loc2 = pixel_loc2[np.all(pixel_loc2>0, axis=1)]
    pixel_loc2 = pixel_loc2.astype(np.int)
    right_out[tuple(pixel_loc2[:,1]), tuple(pixel_loc2[:,0])] = left_in[tuple(pixel_loc[:,1]), tuple(pixel_loc[:,0])].copy()
    
    return right_out



def warp_left_to_right_disparity_fast(left, disparity):
    h,w = disparity.shape[:2]
    
    maps = np.mgrid[0:h,0:w].astype(np.float32)
    mapx_init = maps[1]
    mapy = maps[0]
    
    mapx = mapx_init + disparity
    mapx[mapx>w]=w
    remaped = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
    remaped[disparity==0]=0
    return remaped

if __name__ == "__main__":
    
    # root_dir = Path('/home/dimitrisps/Datasets/UCL-SERV-CT/Rectified')
    # left_rect_dir = root_dir /'Left_rectified'
    # right_rect_dir = root_dir /'Right_rectified_cc'
    # calib_dir = root_dir /'Rectified_calibration'
    # disparity_dir = root_dir /'occ'
    
    # warped_right_clean_dir = root_dir /'warped_left_clean_fast_dir'
    # warped_right_clean_dir.mkdir(parents=True, exist_ok=True)
    
    # root_dir = Path('/home/dimitrisps/new_data')
    # left_rect_dir = root_dir /'rect_left'
    # right_rect_dir = root_dir /'rect_right'
    # calib_dir = root_dir /'complete_calib'
    # depthmap_scared_dir = root_dir /'clean_left_depthmap'
    
    # clean_disparity_dir = root_dir /'clean_disparity_128'
    # clean_disparity_dir.mkdir(parents=True, exist_ok=True)
    # warped_right_clean_dir = root_dir /'warped_right_vec_dir'
    # warped_right_clean_dir.mkdir(parents=True, exist_ok=True)   
    
    # left_paths = sorted([str(p) for p in left_rect_dir.iterdir()])
    # right_paths = sorted([str(p) for p in right_rect_dir.iterdir()])
    # calib_paths = sorted([str(p) for p in calib_dir.iterdir()])
    # depth_paths = sorted([str(p) for p in depthmap_scared_dir.iterdir()])
    
    
    
    
    # for calib_p, left_p, right_p, depth_p in zip(calib_paths, left_paths, right_paths, depth_paths):
    # # for calib_p, left_p, right_p in zip(calib_paths, left_paths, right_paths):        
    #     scared_gt = iotools.load_depthmap_xyz(depth_p)
    #     left_rect = cv2.imread(left_p)
    #     right_rect = cv2.imread(right_p)
    #     calib = StereoCalibrator().load(calib_p)
        
    #     # create clean disparity
        
    #     # disparity_img = iotools.load_subpix_png(disparity_dir/Path(left_p).name)[0]
    #     rot_gt = dm.transform_pts(scared_gt.reshape(-1,3), dm.create_RT(R=calib['R1']))
    #     disparity_img = dm.ptd3d_to_disparity(rot_gt, calib['P1'], calib['P2'], (1024,1280))
    #     iotools.save_subpix_png(clean_disparity_dir / Path(left_p).name, disparity_img, scale_factor=128.0)
        
    #     # warp right to left
    #     start = time.time()
    #     warped_right = warp_left_to_right_disparity_fast(right_rect, -disparity_img.astype(np.float32))
    #     # print(time.time()-start)
    #     cv2.imwrite(str(warped_right_clean_dir / Path(left_p).name), warped_right)
    #     # compute the error metrics
    #     err, s = left_right_consistancy_check(left_rect, warped_right)
    #     print(Path(left_p).name, err, s)
    
    
    root_dir = Path('/home/dimitrisps/Datasets/UCL-SERV-CT/Rectified')
    left_rect_dir = root_dir /'Left_rectified'
    warped_right_clean_dir = root_dir /'warped_left_clean_fast_dir'
    
    left_paths = sorted([str(p) for p in left_rect_dir.iterdir()])
    warped_right_paths = sorted([str(p) for p in warped_right_clean_dir.iterdir()])
    
    print('serv-ct results')
    for left, warped_right, in zip(left_paths, warped_right_paths):

        err, s = left_right_consistancy_check(cv2.imread(left), cv2.imread(warped_right))
        print(Path(left).name, err, s)
    
    
    
    root_dir = Path('/home/dimitrisps/new_data')
    left_rect_dir = root_dir /'rect_left'
    warped_right_clean_dir = root_dir /'warped_left'
    
    left_paths = sorted([str(p) for p in left_rect_dir.iterdir()])
    warped_right_paths = sorted([str(p) for p in warped_right_clean_dir.iterdir()])
    
    print('scared results')
    for left, warped_right, in zip(left_paths, warped_right_paths):

        err, s = left_right_consistancy_check(cv2.imread(left), cv2.imread(warped_right))
        print(Path(left).name, err, s)
        