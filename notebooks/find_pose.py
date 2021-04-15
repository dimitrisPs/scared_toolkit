import cv2
import numpy as np
import sys; sys.path.append('..')
from scareddataset import iotools
import scareddataset.data_maniputlation as dm
from scareddataset.calibrator import StereoCalibrator
from pathlib import Path

data_dir = Path('/home/dimitrisps/Datasets/Scared_stereo/scared/dataset4/keyframe1')

left =  cv2.imread(str(data_dir/'Left_Image.png'))
right = cv2.imread(str(data_dir/'Right_Image.png'))


cv2.imwrite('./left.png', left)
cv2.imwrite('./right.png', right)
pts3d_left = iotools.load_depthmap_xyz(data_dir/'left_depth_map.tiff')
pts3d_right = iotools.load_depthmap_xyz(data_dir/'right_depth_map.tiff')

depthmap_left = dm.scared_to_depthmap(pts3d_left)
depthmap_right = dm.scared_to_depthmap(pts3d_right)

# iotools.export_ply('ds4kf1_left.ply',pts3d_left.reshape(-1,3))
# iotools.export_ply('ds4kf1_right.ply',pts3d_right.reshape(-1,3))
# exit()

iotools.save_subpix_png('./depthmap_left.png', depthmap_left)
iotools.save_subpix_png('./depthmap_right.png', depthmap_right)

calibrator = StereoCalibrator()
calib = calibrator.load(str(data_dir/'endoscope_calibration.yaml'))

left_undistorted = cv2.undistort(left, calib['K1'], calib['D1'])
right_undistorted = cv2.undistort(right, calib['K2'], calib['D1'])



RT = dm.create_RT(R = calib['R'], T=calib['T'])
left_pts_cp = pts3d_left.copy()
left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
iotools.save_subpix_png('./depthmap_right_from_left_origina_calib.png', depthmap_right_to_left)

RT_cc_ds1kf5 = np.asarray([[0.999998927116, -0.001379763242, 0.000936961791, 4.062195301056],
[0.001380137634, 0.999998867512, -0.000394259929, -0.111023575068],
[-0.000936408993, 0.000395551149, 1.000000000000, -0.032032929361],
[0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])


RT_cc_ds4kf1 = np.asarray([[0.999954462051, 0.005962971598, -0.007483757101, 4.680027008057],
[-0.006025682669, 0.999947011471, -0.008385091089, 0.430452585220],
[0.007433366496, 0.008429791778, 0.999937295914, -0.123631983995],
[0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])


left_pts_cp = pts3d_left.copy()
left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),np.linalg.inv(RT_cc_ds4kf1))
depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
iotools.save_subpix_png('./depthmap_right_from_left_ipc_cc.png', depthmap_right_to_left)

print(RT)
print(RT_cc_ds4kf1)
orb = cv2.ORB_create(nfeatures=5000)

kp_left, des_left = orb.detectAndCompute(left_undistorted,None)
kp_right, des_right = orb.detectAndCompute(right_undistorted,None)
# import matplotlib.pyplot as plt
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
# matches = bf.match(des_left,des_right)
# Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# print(matches)
# Draw first 10 matches.
# img3 = cv2.drawMatches(left,kp_left,right,kp_right,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
search_params = dict(checks=500)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des_left.astype(np.float32),des_right.astype(np.float32),k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp_right[m.trainIdx].pt)
        pts1.append(kp_left[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

E = calib['K2'].T@F@calib['K1'];

R1, R2, T = cv2.decomposeEssentialMat(E)

norm_T = calib['T'].reshape(-1)/T.reshape(-1)

# RT = dm.create_RT(R = R1, T=norm_T)
# left_pts_cp = pts3d_left.copy()
# left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
# depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
# iotools.save_subpix_png('./depthmap_right_from_left_R1_T.png', depthmap_right_to_left)

# RT = dm.create_RT(R = R2, T=norm_T)
# left_pts_cp = pts3d_left.copy()
# left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
# depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
# iotools.save_subpix_png('./depthmap_right_from_left_R2_T.png', depthmap_right_to_left)

mag_T = np.sqrt(np.sum(T**2))
mag_calib_T = np.sqrt(np.sum(calib['T']**2))
mag_dif = -mag_calib_T/mag_T
print(T)

T_new = mag_dif *T

RT = dm.create_RT(R = R1, T=calib['T'])
left_pts_cp = pts3d_left.copy()
left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
iotools.save_subpix_png('./depthmap_right_from_left_R1.png', depthmap_right_to_left)

RT = dm.create_RT(R = R2, T=calib['T'])
left_pts_cp = pts3d_left.copy()
left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
iotools.save_subpix_png('./depthmap_right_from_left_R2.png', depthmap_right_to_left)

RT = dm.create_RT(R = R1, T=T_new)
left_pts_cp = pts3d_left.copy()
left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
iotools.save_subpix_png('./depthmap_right_from_left_R1_T.png', depthmap_right_to_left)

for multiplier in np.arange(4,5,0.1):
    T_tmp = -T*multiplier
    print(T_tmp)
    RT = dm.create_RT(R = R2, T=T_tmp)
    left_pts_cp = pts3d_left.copy()
    left_to_right = dm.transform_pts(left_pts_cp.reshape(-1,3),RT)
    depthmap_right_to_left = dm.pts3d_to_depthmap(left_to_right.reshape(-1,3), calib['K2'], calib['D2'], (1024,1280))[0]
    iotools.save_subpix_png('./depthmap_right_from_left_R2_T_{:02.3f}.png'.format(multiplier), depthmap_right_to_left)

print(calib['T'])
print(mag_dif)
# print(T)
# print(calib['T'].reshape(-1)/T.reshape(-1))
