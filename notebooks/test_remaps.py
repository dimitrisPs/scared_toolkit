import numpy as np
import cv2
import sys;sys.path.append('..')
from scareddataset import iotools
import time



def warp_left_to_right(left, disparity):
    h,w = disparity.shape[:2]
    print(h,w)
    
    maps = np.mgrid[0:h,0:w].astype(np.float32)
    print(maps.shape)
    mapx_init = maps[1]
    mapy = maps[0]
    
    mapx = mapx_init + disparity
    mapx[mapx>w]=w
    mapx[disparity==0] =0
    remaped = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
    return remaped
            
if __name__ == "__main__":
    left = cv2.imread('/home/dimitrisps/UCL-SERV-CT/Rectified/Left_rectified/009.png')
    right = cv2.imread('/home/dimitrisps/UCL-SERV-CT/Rectified/Right_rectified_cc/009.png')
    disparity = iotools.load_subpix_png('/home/dimitrisps/UCL-SERV-CT/Rectified/occ/009.png')[0]
    
    
    start = time.time()
    remaped = warp_left_to_right(left, disparity)
    print(time.time()-start)
    
    cv2.imwrite('./warped.png',remaped)
    cv2.imwrite('./warped_left.png', left)
    cv2.imwrite('./warped_right.png', right)