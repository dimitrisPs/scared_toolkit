import cv2
import csv
import numpy as np
from pathlib import Path
import argparse
import sys;sys.path.append('..')
from scareddataset.calibrator import StereoCalibrator
import scareddataset.data_maniputlation as dm
from scareddataset import iotools

def specular_mask(img, lightness_val=200):
    mask =np.zeros(img.shape[:2], dtype=np.uint8)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask[img_hsl[:,:,1]<lightness_val]=255
    return mask

def stereo_anagplyph(left, right):
    height, width = left.shape[:2]
    out = np.zeros((height,width,3), np.uint8)
    #R =left, G=left/2+right/2, B=right
    out[:,:,2]=left.copy()
    out[:,:,0]=right.copy()
    out[:,:,1]= right.copy()
    return out

def overlay_matches(img, left, right):
    anaglyph = img.copy()
    for point_left, point_right in zip(list(left), list(right)):
        # print(point_left, point_right)
        cv2.drawMarker(anaglyph, tuple(point_left), ((255,0,0)), thickness=2, markerSize=25)
        cv2.drawMarker(anaglyph, tuple(point_right), ((0,0,255)), thickness=2, markerSize=25)
        cv2.arrowedLine(anaglyph, tuple(point_left), tuple(point_right), (0,255,0))
    return anaglyph


def import_matches(path):
    left_matches=[]
    right_matches=[]
    with open(path,'r') as csv_file:
        read_worker = csv.reader(csv_file)
        for row in read_worker:
            row = list(map(lambda x: float(x), row))
            left_matches.append(row[:2])
            right_matches.append(row[2:])
    return np.asarray(left_matches), np.asarray(right_matches)

def find_lk_matches(img_l, img_r):
        # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1000,
                        qualityLevel = 1e-10,
                        minDistance = 15,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (7,7),
                    maxLevel = 7,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10))
    img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    
    # extract Shi-Tomasi features on the left rectified image.
    features_left = cv2.goodFeaturesToTrack(img_l_gray, **feature_params, mask=specular_mask(img_l))
    features_right, valid, err = cv2.calcOpticalFlowPyrLK(img_l_gray,
                                                            img_r_gray,
                                                            features_left,
                                                            None,
                                                            **lk_params)
    # keep only matched features
    good_left = features_left[valid==1]
    good_right = features_right[valid==1]
    return good_left, good_right

def RT_from_E(E, calib, left_match, right_match):
    R1, R2, T = cv2.decomposeEssentialMat(E)
    
    P1 = calib['K1'] @ np.eye(4)[:3,:]
    RT_select =[dm.create_RT(R1, T),
                dm.create_RT(R2, T),
                dm.create_RT(R1, -T),
                dm.create_RT(R2, -T)]
    for i, RT in enumerate(RT_select):
        P2 = calib['K2'] @ RT[:3,:]
        pt_3d = cv2.triangulatePoints(P1, P2, left_match.T, right_match.T).T
        pt_3d = pt_3d[:,:3]/(pt_3d[:,3].reshape(-1,1))
        if pt_3d[0,2]<0:
            continue
        pt_3d_rot = dm.transform_pts(pt_3d, RT)
        if pt_3d_rot[0,2]<0:
            continue
        print(i)
        return RT[:3,:3], RT[:3,3].reshape(-1,1)
    return None

def fix_T_mag(R, T, calib, match_left, match_right, depthmap):
    
    P1 = calib['K1'] @ np.eye(4)[:3,:]
    P2 = calib['K2'] @ dm.create_RT(R, T)[:3,:]
    pts_3d = cv2.triangulatePoints(P1,P2, match_left.astype(np.float32).T, match_right.astype(np.float32).T).T
    pts_3d_triang = pts_3d[:,:3]/(pts_3d[:,3].reshape(-1,1))
    pts3d_depthmap = depthmap[tuple(good_left[:,1].astype(np.int)), tuple(good_left[:,0].astype(np.int))]
    pts_3d_triang = pts_3d_triang[~np.isnan(pts3d_depthmap).any(axis=1)]
    pts3d_depthmap = pts3d_depthmap[~np.isnan(pts3d_depthmap).any(axis=1)]
    mag_1= np.sqrt(np.sum(pts_3d_triang**2,axis=1))
    mag_2 = np.sqrt(np.sum(pts3d_depthmap**2,axis=1))
    scale = (np.median(mag_1/mag_2))
    return T/scale


def drawlines(left,right,lines,left_pts,right_pts, display_interval=None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    h,w = left.shape[:2]
    lines=lines.reshape(-1,3)
    if display_interval is not None:
        lines=lines[::display_interval]
        left_pts=left_pts[::display_interval]
        right_pts=right_pts[::display_interval]
    for r,pt1,pt2 in zip(lines,left_pts,right_pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        img1 = cv2.line(left, (x0,y0), (x1,y1), color,1)
        img1 = cv2.drawMarker(left,tuple(pt1),color, markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
        img2 = cv2.drawMarker(right,tuple(pt2),color,markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    return img1,img2
    
    


if __name__ == "__main__":
    
    frame_id='4_2'
    
    root_dir = Path('/home/dimitrisps/new_data')
    left_original_path = root_dir /'left'/(frame_id +'.png')
    right_original_path = root_dir /'right'/(frame_id +'.png')
    left_depthmap_path = root_dir /'clean_left_depthmap'/(frame_id +'.tiff')
    calib_path = root_dir /'original_calib'/(frame_id +'.yaml')
    
    calibrator = StereoCalibrator()
    calib = calibrator.load(str(calib_path))
    
    left_img = cv2.imread(str(left_original_path))
    right_img = cv2.imread(str(right_original_path))
    
    
    left_img_undistort = cv2.undistort(left_img,calib['K1'], calib['D1'])
    right_img_undistort = cv2.undistort(right_img, calib['K2'], calib['D2'])
    left_depthmap_scared = iotools.load_depthmap_xyz(left_depthmap_path)
    
    
    #original matrices.
    rect_left, rect_right = calibrator.rectify(left_img, right_img, -1)
    cv2.imwrite('./rect_old_left.png', rect_left)
    cv2.imwrite('./rect_old_right.png', rect_right)
    good_left, good_right = find_lk_matches(rect_left, rect_right)
    F, mask = cv2.findFundamentalMat(good_left,good_right,cv2.FM_RANSAC, ransacReprojThreshold=0.02,confidence=0.99)
    mask = mask.squeeze()
    good_left = good_left[mask==1]
    good_right = good_right[mask==1]
    
    anaglyph = stereo_anagplyph(cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY))
    anaglyph_matches = overlay_matches(anaglyph, good_left, good_right)
    cv2.imwrite('./anaglyph_rect_old_R.png', anaglyph_matches)
    lines = cv2.computeCorrespondEpilines(good_left,1, F)
    epi_right, epi_left = drawlines(rect_right, rect_left, lines, good_right, good_left, display_interval=5)
    cv2.imwrite('./epilines_old.png', np.hstack((epi_left, epi_right)))

    
    
    

    # compute autocalib.
    good_left, good_right = find_lk_matches(left_img, right_img)
    F, mask = cv2.findFundamentalMat(good_left,good_right,cv2.FM_RANSAC, ransacReprojThreshold=0.2,confidence=0.99)
    E = calib['K2'].T@F@calib['K1']
    RT = RT_from_E(E, calib, good_left, good_right)
    if RT is None:
        print('check_code')
        exit()
    R, T= RT
    
    T = fix_T_mag(R, T, calib, good_left, good_right, left_depthmap_scared)
    calibrator = StereoCalibrator()
    calib = calibrator.load(str(calib_path))
    print(calib['T'])
    calib['R'] = R
    calib['T'] = T
    print(calib['T'])

    rect_left, rect_right = calibrator.rectify(left_img, right_img, -1)
    cv2.imwrite('./rect_new_left.png', rect_left)
    cv2.imwrite('./rect_new_right.png', rect_right)
    
    good_left, good_right = find_lk_matches(rect_left, rect_right)
    
    F, mask = cv2.findFundamentalMat(good_left,good_right,cv2.FM_RANSAC, ransacReprojThreshold=0.05,confidence=0.999)
    mask = mask.squeeze()
    good_left = good_left[mask==1]
    good_right = good_right[mask==1]
    
    anaglyph = stereo_anagplyph(cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY))
    anaglyph_matches = overlay_matches(anaglyph, good_left, good_right)
    cv2.imwrite('./anaglyph_rect_new_R.png', anaglyph_matches)
    lines = cv2.computeCorrespondEpilines(good_left,1, F)
    epi_right, epi_left = drawlines(rect_right, rect_left, lines, good_right, good_left, display_interval=5)
    cv2.imwrite('./epilines_new.png', np.hstack((epi_left, epi_right)))
    
    
    #stereo rectify uncalibrated.
    left_img_undistort = cv2.undistort(left_img,calib['K1'], calib['D1'])
    right_img_undistort = cv2.undistort(right_img, calib['K2'], calib['D2'])

    
    good_left, good_right = find_lk_matches(left_img_undistort, right_img_undistort)
    
    F, mask = cv2.findFundamentalMat(good_left,good_right,cv2.FM_RANSAC, ransacReprojThreshold=0.02,confidence=0.999)
    
    ret, H1,H2 = cv2.stereoRectifyUncalibrated(good_left, good_right, F,(1280,1024))
    rect_left = cv2.warpPerspective(left_img_undistort, H1,(1280,1024))
    rect_right = cv2.warpPerspective(right_img_undistort, H2,(1280,1024))
    cv2.imwrite('./rect_unc_left.png', rect_left)
    cv2.imwrite('./rect_unc_right.png', rect_right)
    
    
    good_left, good_right = find_lk_matches(rect_left, rect_right)
    
    F, mask = cv2.findFundamentalMat(good_left,good_right,cv2.FM_RANSAC, ransacReprojThreshold=0.02,confidence=0.999)

    mask = mask.squeeze()
    good_left = good_left[mask==1]
    good_right = good_right[mask==1]
    
    anaglyph = stereo_anagplyph(cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY))
    anaglyph_matches = overlay_matches(anaglyph, good_left, good_right)
    cv2.imwrite('./anaglyph_rect_unc_R.png', anaglyph_matches)
    lines = cv2.computeCorrespondEpilines(good_left,1, F)
    epi_right, epi_left = drawlines(rect_right, rect_left, lines, good_right, good_left, display_interval=5)
    cv2.imwrite('./epilines_unc.png', np.hstack((epi_left, epi_right)))
    
    # plot epilines
