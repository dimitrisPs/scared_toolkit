import numpy as np
import cv2


def depthmap_to_pts3d(depthmap, K, D=np.zeros((5,1))):
    """Converts one depthmap to xyz image.
    
    Given a Depthmap this functions reprojects points to 3d using the calibration
    parameteres provided. for a location (u,v), a depthmap stores the length of
    the vector starting from the camera center and ending to a point in 3D which
    projects to (u,v). Using the camera parameteres we can get the 3d direction
    of those vectors and then normalize their length accorind to the depthmap
    in order to compute the 3d geometry captured in the image. the function 
    outputs this information as a 3 channel image, with each channel representing
    a 3d component, namely X,Y,Z in this order.
    Args:
        depthmap (np.ndarray): depthamp image with dimentrion of hxw
        K (np.ndarray): Camera matrix 
        D (np.ndarray): Camera Distortion coefficients

    Returns:
        np.ndarray: xyz image of dimentions hxwx3
    """
    #create a with pixel locations as values vectorsize it and make it pixel homogeneous
    h,w = depthmap.shape[:2]
    pixel_loc = np.mgrid[0:w,0:h].transpose(2,1,0).astype(np.float)
    pixel_loc=pixel_loc.reshape(-1,2)
    
    # project pixels in the image
    image_plane_pts = cv2.undistortPoints(pixel_loc, K, D).squeeze()
    image_plane_pts_h = np.hstack((image_plane_pts, np.ones((image_plane_pts.shape[0],1))))
    
    #normalize and multiply by depth
    norm = np.sqrt(np.sum(image_plane_pts_h**2,axis=1)).reshape(-1,1)
    image_plane_pts_h_norm=image_plane_pts_h / norm
    xyz_map = image_plane_pts_h_norm * depthmap.reshape(-1,1)

    return xyz_map.reshape(h,w,3)


def pts3d_to_depthmap(pts3d):
    """covert 3 channel xyz image to 1 channel depthmap

    Args:
        pts3d (np.ndarray): hxwx3 xyz image, each image point encodes the 3d
        location of the point is the projection of.

    Returns:
        np.ndarray: hxw depthmap. each element is the length of the vector 
        starting from camera center and end up in a point in 3d. Each such vector
        passes through a pixel in image plane. 
    """
    h, w, c = pts3d.shape
    pts3d = pts3d.reshape(-1,3)
    depthmap = np.sqrt(np.sum(pts3d**2, axis=1))
    return depthmap.reshape(h,w)



def disparities_to_pts3d(disparity, calib):
    pass
    