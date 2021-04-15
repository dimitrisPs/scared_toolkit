from pathlib import Path
from shutil import copyfile
from scareddataset.data_maniputlation import scared_to_depthmap, transform_pts, create_RT, ptd3d_to_disparity
from scareddataset.iotools import save_subpix_png, load_depthmap_xyz
from scareddataset.calibrator import StereoCalibrator
import cv2
import numpy as np


def warp_left_to_right_disparity(left_in, disparity):
    # start = time.time()
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


root_data_dir = Path("/home/dimitrisps/Datasets/Scared_stereo/scared")
# root_data_dir = Path("/home/dimitrisps/Datasets/Scared_stereo/test_data")
dst_dir = Path("/home/dimitrisps/Datasets/Scared_stereo/train_keyframes")

dst_left_depthmaps_dir = dst_dir/"left_depthmaps"
dst_right_depthmaps_dir = dst_dir/"right_depthmaps"

dst_left_img_dir = dst_dir/"left"
dst_right_img_dir= dst_dir/"right"

dst_left_rect_img_dir = dst_dir/"rect_left"
dst_right_rect_img_dir= dst_dir/"rect_right"

dst_disparity_dir = dst_dir/"disparity"

dst_calib_dir=dst_dir/"calib"

rect_left_warp_to_right_dir=dst_dir/"rect_left_warp_to_right"

rect_left_warp_to_right_dir.mkdir(parents=True, exist_ok=True)
dst_calib_dir.mkdir(parents=True, exist_ok=True)
dst_disparity_dir.mkdir(parents=True, exist_ok=True)
dst_left_rect_img_dir.mkdir(parents=True, exist_ok=True)
dst_right_rect_img_dir.mkdir(parents=True, exist_ok=True)
dst_left_depthmaps_dir.mkdir(parents=True, exist_ok=True)
dst_right_depthmaps_dir.mkdir(parents=True, exist_ok=True)
dst_left_img_dir.mkdir(parents=True, exist_ok=True)
dst_right_img_dir.mkdir(parents=True, exist_ok=True)


for dataset in root_data_dir.iterdir():
    for keyframe in dataset.iterdir():
        
        
        output_name = str(dataset)[-1]+"_"+str(keyframe)[-1]+".png"
        output_calib_name =  str(dataset)[-1]+"_"+str(keyframe)[-1]+".json"
        depthmap_left_path = keyframe/"left_depth_map.tiff"
        depthmap_right_path = keyframe/"right_depth_map.tiff"
        
        scared_depthmap_left = load_depthmap_xyz(depthmap_left_path)
        scared_depthmap_right = load_depthmap_xyz(depthmap_right_path)
    
        depthmap_left_png = scared_to_depthmap(scared_depthmap_left)
        depthmap_right_png = scared_to_depthmap(scared_depthmap_right)
        
        save_subpix_png(dst_left_depthmaps_dir/output_name, depthmap_left_png)
        save_subpix_png(dst_right_depthmaps_dir/output_name, depthmap_right_png)
        
        copyfile(keyframe/"Left_Image.png", dst_left_img_dir/output_name)
        copyfile(keyframe/"Right_Image.png", dst_right_img_dir/output_name)
        
        #generate complete calib files, rectify frames and gt disparities.
        
        stereo_calibrator = StereoCalibrator()
        stereo_calibrator.load(str(keyframe/"endoscope_calibration.yaml"))

        left = cv2.imread(str(keyframe/"Left_Image.png"))
        right = cv2.imread(str(keyframe/"Right_Image.png"))
        if left is None: raise FileNotFoundError
        if right is None: raise FileNotFoundError
        
        rect_left, rect_right = stereo_calibrator.rectify(left, right)


        cv2.imwrite(str(dst_left_rect_img_dir/output_name), rect_left)
        cv2.imwrite(str(dst_right_rect_img_dir/output_name), rect_right)
        stereo_calibrator.save(dst_calib_dir / output_calib_name)
        
        # generate disparity 
        calib = stereo_calibrator.calib
        
        size = scared_depthmap_left.shape[:2]


        # rotate gt by R1 to allign it with the rectified left frame
        xyz = scared_depthmap_left.reshape(-1, 3)
        rot_xyz = transform_pts(xyz, create_RT(R=calib['R1']))

        disparity_img = ptd3d_to_disparity(rot_xyz, calib['P1'], calib['P2'], size)

        save_subpix_png(dst_disparity_dir/output_name, disparity_img)
        
        
        
        #warp left to right based on disparity
        
        
        warped = warp_left_to_right_disparity(rect_left, disparity_img)
        cv2.imwrite(str(rect_left_warp_to_right_dir/output_name), warped)