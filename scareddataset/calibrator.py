import cv2
import argparse
import numpy as np


class Calibrator:
    def __init__(self, chessboard_size, tile_size):
        self.chessboard_size = chessboard_size
        self.tile_size = tile_size
        self.calib = {'cb_size': self.chessboard_size,
                      'tile_size': self.tile_size}

    def _create_chessboard_points(self):
                # create 3d points of calibration matrix
        objp = np.zeros(
            (self.chessboard_size[0]*self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        return objp * self.tile_size

    def _find_chessboard_corners(self, img):
        criteria = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_COUNT, 100, 0.0001)
        # check if not gray already TODO
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            return corners

    # TODO:validate paths
    def save(self, path):
        fs_write = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        for k in self.calib.keys():
            if self.calib[k] is not None:
                fs_write.write(k, self.calib[k])
        fs_write.release()

    def load(self, path):
        fs_read = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        # TODO:check is file exist
        for k in self.calib.keys():
            if k =='error':
                continue
            elif k =='K1':
                self.calib[k] = fs_read.getNode('K1').mat()
                if self.calib[k] is None:
                    self.calib[k] = fs_read.getNode('M1').mat()
            elif k =='K2':
                self.calib[k] = fs_read.getNode('K2').mat()
                if self.calib[k] is None:
                    self.calib[k] = fs_read.getNode('M2').mat()
            else:
                self.calib[k] = fs_read.getNode(k).mat()
        fs_read.release()
        return self.calib


class MonoCalibrator(Calibrator):
    def __init__(self, img_path_list, chessboard_size, tile_size):
        calib = {}
        self.chessboard_size = chessboard_size
        self.img_paths = img_path_list
        self.tile_size = tile_size
        self.obj_pts = self._create_chessboard_points()
        self.corners_3d = []
        self.corners_2d = []
        self.img_size = None
        self.calib = {'error': None, 'K': None,
                      'D': None, 'rvecs': None, 'tvecs': None}
        # TODO add img size, tilesize

    def calibrate(self):

        for img_path in self.img_paths:
            sample = cv2.imread(str(img_path))
            if sample is None:
                raise ValueError(str(img_path) + ' is not an image')
            corners = self._find_chessboard_corners(sample)
            if corners is None:
                print("Could not detect patern for image" + str(img_path))
                continue
            self.corners_3d.append(self.obj_pts)
            self.corners_2d.append(corners)

        print(cv2.imread(str(self.img_paths[0])).shape)
        self.img_size = cv2.imread(str(self.img_paths[0])).shape[:2]
        print('calibrating...')
        reprojection_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.corners_3d, self.corners_2d, self.img_size[::-1], None, None)
        print('done\t rms reprojection error: ' + str(reprojection_error))
        self.calib = {'error': reprojection_error, 'K': K, 'D': D,
                      'rvecs': np.asarray(rvecs), 'tvecs': np.asarray(tvecs)}

    def undistort(self, img):
        print("Not implemented")
        return None


class StereoCalibrator(Calibrator):
    def __init__(self, left_img_paths=[], right_img_paths=[], chessboard_size=[], tile_size=None):
        calib = {}
        self.chessboard_size = chessboard_size
        self.left_img_paths = left_img_paths
        self.right_img_paths = right_img_paths
        self.tile_size = tile_size
        if tile_size is not None:
            self.obj_pts = self._create_chessboard_points()
        self.corners_3d = []
        self.left_corners_2d = []
        self.right_corners_2d = []
        self.calib_left = None
        self.calib_right = None
        self.img_size = None
        self.left_rect_map = None
        self.left_rect_map = None
        self.calib = {'error': None, 'image_size': None, 'K1': None, 'D1': None,
                      'K2': None, 'D2': None, 'R': None, 'T': None,
                      'E': None, 'F': None, 'R1': None, 'R2': None, 'P1': None,
                      'P2': None, 'Q': None, 'roi1': None, 'roi2': None}
        # TODO add img size, tilesize

    def calibrate(self):

        # calibrate left and right
        self.calib_left = MonoCalibrator(self.left_img_paths,
                                         self.chessboard_size, self.tile_size)
        self.calib_left.calibrate()

        self.calib_right = MonoCalibrator(self.right_img_paths,
                                          self.chessboard_size, self.tile_size)
        self.calib_right.calibrate()

        self.img_size = self.calib_left.img_size

        # find frames that chessboard is visible from both left and right

        pair_paths = list(zip(self.left_img_paths, self.right_img_paths))

        for left_path, right_path in pair_paths:
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))

            left_corners = self._find_chessboard_corners(left_img)
            right_corners = self._find_chessboard_corners(right_img)

            if (left_corners is not None) and (right_corners is not None):
                # marker visible from both views
                self.left_corners_2d.append(left_corners)
                self.right_corners_2d.append(right_corners)
                self.corners_3d.append(self.obj_pts)
            else:
                print('skipping sample ' + str(left_path.name))

        # calibrate stereo
        error_rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(self.corners_3d,
                                                                    self.left_corners_2d,
                                                                    self.right_corners_2d,
                                                                    self.calib_left.calib['K'],
                                                                    self.calib_left.calib['D'],
                                                                    self.calib_right.calib['K'],
                                                                    self.calib_right.calib['D'],
                                                                    self.img_size,
                                                                    flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        print('RMS reprojection Error: ', error_rms)


        # Store matrices to calib dictionary

        self.calib = {'error': error_rms, 'image_size': self.img_size, 'K1': K1, 'D1': D1,
                      'K2': K2, 'D2': D2, 'R': R, 'T': T, 'E': E, 'F': F}
        self._compute_rectification_parameteres()
    
    def _compute_rectification_parameteres(self, alpha=-1):
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.calib['K1'],
                                                        self.calib['D1'],
                                                        self.calib['K2'],
                                                        self.calib['D2'],
                                                        self.calib['image_size'],
                                                        self.calib['R'].astype(np.float64),
                                                        self.calib['T'].astype(np.float64).reshape(3,1),
                                                        alpha=alpha)
        
        rect_calib = {'R1': R1,'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                      'roi1': roi1, 'roi2': roi2, 'rect_alpha':alpha}
        self.calib.update(rect_calib)

    def rectify(self, left, right):

        if self.left_rect_map is None:
            self.calib['image_size']=left.shape[:2]
            if self.calib['R1'] is None:
                self._compute_rectification_parameteres()
            self.left_rect_map = cv2.initUndistortRectifyMap(self.calib['K1'],
                                                             self.calib['D1'],
                                                             self.calib['R1'],
                                                             self.calib['P1'],
                                                             self.calib['image_size'][::-1],
                                                             cv2.CV_32FC1)
            self.right_rect_map = cv2.initUndistortRectifyMap(self.calib['K2'],
                                                              self.calib['D2'],
                                                              self.calib['R2'],
                                                              self.calib['P2'],
                                                              self.calib['image_size'][::-1],
                                                              cv2.CV_32FC1)

        left_rect = cv2.remap(left, self.left_rect_map[0],
                              self.left_rect_map[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.right_rect_map[0],
                               self.right_rect_map[1], cv2.INTER_LINEAR)
        return left_rect, right_rect

    # TODO:write functions for visualisation
