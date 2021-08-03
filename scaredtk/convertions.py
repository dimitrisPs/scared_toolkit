import numpy as np
import cv2
from typing import Tuple


def disparity_to_img3d(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Convert disparity to 3D image
    
    Convert disparity to 3D image format, similar to what is used to store
    ground truth information in scared. The resulting 3D image is expressed in
    the same frame of reference with the disparity, thus it cannot directly used
    to create 3D images suitable for evaluation on the provided sequence. The 
    unprojection is done using the Q matrix computed during the stereo 
    calibration and rectification phase.

    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray): Q matrix computed during stereo calibration and 
        rectification phase.

    Returns:
        np.ndarray: HxWx3 img3d output array. Each pixel location (u,v) encodes
        the 3D coordinates of the 3D point that projects to u,v.
    """
    assert disparity.dtype == np.float32
    disparity = np.nan_to_num(disparity)
    valid = disparity >= 0
    img3d = cv2.reprojectImageTo3D(disparity, Q)
    img3d[~valid] = np.nan
    return img3d


def disparity_to_ptcloud(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Converts disparity maps to a pointclouds
    
    Converts a disparity image to an N element pointcloud represented as a Nx3
    array. The disparity image is reprojected in 3D using the Q matrix computed
    during the calibration and rectification phase and then removed points whose
    values are inf or nan. 

    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray):  Q matrix computed during stereo calibration and 
        rectification phase.

    Returns:
        np.ndarray: N element pointcloud stored as a Nx3 array.
    """
    assert disparity.dtype == np.float32
    img3d = disparity_to_img3d(disparity, Q)
    return img3d_to_ptcloud(img3d)


def disparity_to_depthmap(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """ Convert disparity to depthmap
    
    Converts a disparity image to depthmap using the provided Q matrix copmuted
    during the stereo calibration and rectification phase. Q is used to 
    unproject the disparity image and constact a 3D image. This function returns
    the 3rd channel of this 3D image which is defined as the depthmap. The output
    of this function can not be used evaluate on the original SCARED ground truth
    as the former is expressed in a stereo rectified frame of reference while the
    former is express in the original frame of reference including distortions.

    Args:
        disparity (np.ndarray): HxW disparity map float array
        Q (np.ndarray):  Q matrix computed during stereo calibration and 
        rectification phase.

    Returns:
        np.ndarray: HxW depthmap float array represented in the same frame of
        reference as the input disparity image.
    """
    img3d = disparity_to_img3d(disparity, Q)
    return img3d_to_depthmap(img3d)


def ptcloud_to_disparity(
    pt_cloud: np.ndarray, P1: np.ndarray, P2: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Converts point clouds to disparity
    
    Convert point cloud to disparity maps with subpixel accuracy. The function
    takes the provided point cloud and project it two stereo rectified views 
    based on the projection matrices P1, P2 which describe the projection to the
    left and right rectified frames of reference respectively. projection
    pixel coordinates in the disparity image are defined as the rounded 
    projection coordinates of the point cloud to the left frame of reference.
    Disparity is defined as the horizontal difference of the projection of a 
    between the left and right rectified frame of defence. 

    Args:
        pt_cloud (np.ndarray): N element pointcloud stored as a Nx3 array 
        P1 (np.ndarray): Projection matrix of the left rectified frame of reference
        P2 (np.ndarray): Projection matrix of the right rectified frame of reference
        size (Tuple[int, int]): height, width of the resulting disparity image.

    Returns:
        np.ndarray: HxW disparity float disparity array.
    """

    h, w = size
    disparity = np.zeros(size)
    projection_l = project_pts(pt_cloud, P1).reshape(-1, 2)
    projection_r = project_pts(pt_cloud, P2).reshape(-1, 2)
    disparities = (projection_l - projection_r)[:, 0]
    # find all points that project inside the image domain.
    projection_l = np.round(projection_l)
    valid_indexes = (
        (projection_l[:, 0] >= 0)
        & (projection_l[:, 0] < w)
        & (projection_l[:, 1] >= 0)
        & (projection_l[:, 1] < h)
    )
    disparity_idxs = projection_l[valid_indexes].astype(int)
    valid_disparities = disparities[valid_indexes]
    xs, ys = disparity_idxs[:, 0], disparity_idxs[:, 1]
    disparity[ys, xs] = valid_disparities
    return disparity


def ptcloud_to_img3d(
    ptcloud: np.ndarray, K: np.ndarray, D: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """converts a pointcloud to 3DImage
    
    Converts a pointcloud to a 3 channel 3D image format, similar to what is 
    used to store ground truth information in SCARED. The resulting 3D image is 
    expressed in the same frame of reference with the pointcloud, thus if the
    point cloud is not expressed in the original frame of reference, the output
    of this function can be used to evaluate on the reference data. Each point 
    of the pointcloud is projected to the the image frame based on the calibration
    parameters and distortions are also supported.

    Args:
        ptcloud (np.ndarray): N element pointcloud represented as a Nx3 array
        K (np.ndarray): Camera matrix of the target projection view
        D (np.ndarray): Distortion coefficients of the target projection view
        size (Tuple[int, int]): Height, Width of the resulting 3D image.

    Returns:
        np.ndarray: HxWx3 3D Image, each pixel encodes the projection location 
        of the point it stores as a 3D vector.
    """
    h, w = size
    img3d = np.full((h, w, 3), fill_value=np.nan)

    projection_coordinates = cv2.projectPoints(ptcloud, np.eye(3), np.zeros(3), K, D)[
        0
    ].squeeze()

    # get the projection coordinates, round them and check which of the points
    # end up within the image view.
    projection_coordinates = np.round(projection_coordinates)
    valid_projection_indexes = (
        (projection_coordinates[:, 0] >= 0)
        & (projection_coordinates[:, 0] < w)
        & (projection_coordinates[:, 1] >= 0)
        & (projection_coordinates[:, 1] < h)
    )
    projection_coordinates = projection_coordinates[valid_projection_indexes].astype(
        int
    )
    projected_3dpoints = ptcloud[valid_projection_indexes]

    xs, ys = projection_coordinates[:, 0], projection_coordinates[:, 1]

    img3d[ys, xs] = projected_3dpoints
    return img3d


def ptcloud_to_depthmap(
    ptcloud: np.ndarray, K: np.ndarray, D: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Convert pointcloud to depthmap

    Converts a pointcloud to a depthamp. The function first creates a 3D image
    based on the provided calibration parameters return the last channel. Depthmaps
    are expressed in the same frame of reference with the pointcloud, thus if the
    point cloud is not expressed in the original frame of reference, the output
    of this function can be used to evaluate on the reference data. Each point 
    of the pointcloud is projected to the the image frame based on the calibration
    parameters and distortions are also supported.
    
    
    Args:
        ptcloud (np.ndarray): N element pointcloud represented as a Nx3 array
        K (np.ndarray): Camera matrix of the target projection view
        D (np.ndarray): Distortion coefficients of the target projection view
        size (Tuple[int, int]): Height, Width of the resulting depthmap.

    Returns:
        np.ndarray: HxW float depthmap
    """
    img3d = ptcloud_to_img3d(ptcloud, K, D, size)
    return img3d_to_depthmap(img3d)


def img3d_to_ptcloud(img3d: np.ndarray) -> np.ndarray:
    """Convert 3D image to pointcloud. 
    
    Converts 3D image to pointcloud by reshaping the 3DImage and removing
    unknown points which are stored as nan values in the input 3D image.

    Args:
        img3d (np.ndarray): HxWx3 array, each pixel location (u,v) encodes
        the 3D coordinates of the a point that projects to u,v.

    Returns:
        np.ndarray: N element pointcloud stored as Nx3 array.
    """

    ptcloud = img3d.copy().reshape(-1, 3)

    return ptcloud[~np.isnan(ptcloud).any(axis=1)]


def img3d_to_depthmap(img3d: np.ndarray) -> np.ndarray:
    """Converts 3D image to depthmap
    
    Convertion is simple because the depthmap is the 3rd channel of a 3D image.

    Args:
        img3d (np.ndarray): HxWx3 array, each pixel location (u,v) encodes
        the 3D coordinates of the a point that projects to u,v.


    Returns:
        np.ndarray: HxW float depthmap
    """
    return img3d[:, :, 2].copy()


def img3d_to_disparity(img3d: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Convert 3D image to disparity
    
    Convert 3D Images to disparity maps with subpixel accuracy. The function
    takes the provided 3D Image and project it two stereo rectified views 
    based on the projection matrices P1, P2 which describe the projection to the
    left and right rectified frames of reference respectively. Projection
    pixel coordinates in the disparity image are defined as the rounded 
    projection coordinates of the 3D points to the left frame of reference.
    Disparity is defined as the horizontal difference of the projection of a 
    between the left and right rectified frame of defence. 

    Args:
        img3d (np.ndarray): HxWx3 array, each pixel location (u,v) encodes
        the 3D coordinates of the a point that projects to u,v.
        P1 (np.ndarray): Projection matrix of the left rectified frame of reference
        P2 (np.ndarray): Projection matrix of the right rectified frame of reference

    Returns:
        np.ndarray: HxW disparity float disparity array.
    """
    h, w = img3d.shape[:2]

    ptcloud = img3d_to_ptcloud(img3d)
    return ptcloud_to_disparity(ptcloud, P1, P2, (h, w))


def depthmap_to_ptcloud(
    depthmap: np.ndarray, K: np.ndarray, D: np.ndarray = np.zeros(5)
) -> np.ndarray:
    """Convert depthmap to pointcloud
    
    Convert depthmap to pointcloud using the calibration parameters. The function
    models distortions as well.

    Args:
        depthmap (np.ndarray): HxW depthmap float array
        K (np.ndarray): Camera matrix of source depthmap
        D (np.ndarray, optional):Distortion coefficients of source depthmap
        view.Defaults to np.zeros(5).


    Returns:
        np.ndarray: N element pointcloud stored as a Nx3 array.
    """
    img3d = depthmap_to_img3d(depthmap, K, D)
    return img3d_to_ptcloud(img3d)


def depthmap_to_img3d(
    depthmap: np.ndarray, K: np.ndarray, D: np.ndarray = np.zeros((5, 1))
) -> np.ndarray:
    """Converts one depthmap to xyz image.
    

    Args:
        depthmap (np.ndarray): HxW depthmap
        K (np.ndarray): Camera matrix of source and dest
        D (np.ndarray): Camera Distortion  of source and dest.Defaults to
        np.zeros(5).

    Returns:
        np.ndarray: HxWx3 3D Image.
    """
    # create a with pixel locations as values vectorsize it and make it pixel homogeneous

    # depthmap = np.nan_to_num(depthmap)
    h, w = depthmap.shape[:2]
    pixel_loc = np.mgrid[0:w, 0:h].transpose(2, 1, 0).astype(np.float64)
    pixel_loc = pixel_loc.reshape(-1, 2)

    # project pixels to the image plane. Because we do not provide a new camera
    # matrix, the opencv function returns points in normalized coordinates which
    # can be used to contruct the homogeneous representation of those points.
    image_plane_pts = cv2.undistortPoints(pixel_loc, K, D).squeeze()

    # express normlaized points in homogeneous coordinates.
    image_plane_pts_h = np.hstack(
        (image_plane_pts, np.ones((image_plane_pts.shape[0], 1)))
    )

    # scale the homogeneous coordinates based on depthmap values to get 3D
    # locations.
    img3d = image_plane_pts_h * depthmap.reshape(-1, 1)

    img3d = img3d.reshape(h, w, 3)

    # replace ambiguous values with nan.
    img3d[np.isnan(depthmap)] = [np.nan, np.nan, np.nan]

    return img3d


def depthmap_to_disparity(depthmap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Convert depthmap to disparity
    
    Convert a depthmap to disparity expressed in the same frame of reference.
    This function can only be used correctly when depthmaps are expressed in the
    left stereo rectified frame of reference. The convertion is done using the
    Q matrix computed during the stereo calibration and rectification phase.
    Because Q encodes the relationship between disparity and depthmap we can 
    use it to reproject the disparity map instead of the disparity and keep only
    Z dimension, effectively creating a disparity from a depthmap.

    Args:
        depthmap (np.ndarray): HxW Depthmap
        Q (np.ndarray): Q matrix computed during stereo calibration and 

    Returns:
        np.ndarray: HxW disparity map float array
    """
    return cv2.reprojectImageTo3D(depthmap, Q)[:, :, 2]


def project_pts(pts3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """project 3d points to image, according to projection matrix P

    Args:
        pts3d (np.ndarray): Nx3 array containing 3d points
        P (np.ndarray): projection matrix

    Returns:
        np.ndarray: Nx2 array containing pixel coordinates of projected points.
    """
    # convert to homogeneous
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    projected_pts = (P @ pts3d_h.T).T
    # convert from homogeneous coordinates.
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2].reshape(-1, 1)
    return projected_pts


def transform_pts(pts3d: np.ndarray, RT: np.ndarray) -> np.ndarray:
    """transform points using RT homogeneous matrix

    Args:
        pts3d (np.ndarray): Nx3 array containing 3d point coordinates
        RT (np.ndarray): 4x4 homogeneous transformation matrix

    Returns:
        np.ndarray: Nx3 transformed pts3d points according to RT
    """
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    rotated_pts3d_h = (RT @ pts3d_h.T).T
    rotated_pts3d = rotated_pts3d_h[:, :3] / (rotated_pts3d_h[:, 3].reshape(-1, 1))
    return rotated_pts3d


def create_RT(R: np.ndarray = np.eye(3), T: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix

    Args:
        R (np.ndarray, optional): 3x3 rotation matrix. Defaults to np.eye(3).
        T (np.ndarray, optional): translation vector. Defaults to np.zeros(3).

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix
    """
    RT = np.eye(4)
    RT[:3, :3] = R.copy()
    RT[:3, 3] = T.reshape(3).copy()
    return RT


def ptcloud_to_flow(
    pt_cloud: np.ndarray,
    pose_1: np.ndarray,
    pose_2: np.ndarray,
    size: Tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Generate flow map between pose_1 and pose_2 based on pt_cloud

    Args:
        pt_cloud (np.ndarray): N element pointcloud represented as a Nx3 array
        pose_1 (np.ndarray): 4x4 homogeneous transformation matrix describing 
        how to transform the pt_cloud to the pose at t=0
        pose_2 (np.ndarray): 4x4 homogeneous transformation matrix describing 
        how to transform the pt_cloud to the pose at t=t+1
        size (Tuple[int, int]): Height, Width of the resulting flow map.
        K (np.ndarray): Camera matrix of the target projection view
        D (np.ndarray): Distortion coefficients of the target projection view
        

    Returns:
        np.ndarray: [description]
    """
    h, w = size
    # channel sequence: u,v, nan values where we do not have flow info
    forward_flow = np.full((h, w, 2), fill_value=np.nan)

    # tranform points in space according to the provided kinematics
    pt_cloud1 = transform_pts(pt_cloud, pose_1)
    pt_cloud2 = transform_pts(pt_cloud, pose_2)

    # project the rotated pointclouds into two consecutive frames
    projection_coordinates1 = cv2.projectPoints(
        pt_cloud1, np.eye(3), np.zeros(3), K, D
    )[0].squeeze()
    projection_coordinates2 = cv2.projectPoints(
        pt_cloud2, np.eye(3), np.zeros(3), K, D
    )[0].squeeze()

    # measure projection displacement(flow) in decimal value
    pixel_displacement = projection_coordinates2 - projection_coordinates1

    # compute pixel location the flow values are going to be stored(integer)
    projection_coordinates = np.round(projection_coordinates1)
    valid_projection_indexes = (
        (projection_coordinates[:, 0] >= 0)
        & (projection_coordinates[:, 0] < w)
        & (projection_coordinates[:, 1] >= 0)
        & (projection_coordinates[:, 1] < h)
    )
    # WARNING because i compute the displacement before rounding the projection coordinates
    # i introduce a small error of 0.5 pixels at most.

    projection_coordinates = projection_coordinates1[valid_projection_indexes].astype(
        int
    )
    visible_flow = pixel_displacement[valid_projection_indexes]

    xs, ys = projection_coordinates[:, 0], projection_coordinates[:, 1]

    forward_flow[ys, xs] = visible_flow
    return forward_flow
