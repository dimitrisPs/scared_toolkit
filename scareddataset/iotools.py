import numpy as np
import cv2
import tifffile as tiff
import warnings

from pathlib import Path
import errno
import os

import plyfile


def load_depthmap_xyz(path):
    """loads depthmap in the original f

    Args:
        path ([str, pathlib.Path]): Path to depthmap file.

    Raises:
        FileNotFoundError: When depthmap_path does not point to an existing file.

    Returns:
        np.ndarray: depthmap data.
    """

    depthmap_path_p = Path(path).resolve()
    if not depthmap_path_p.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                path)
    depthmap = tiff.imread(str(depthmap_path_p))
    depthmap[depthmap==0]=np.nan
    return depthmap.astype(np.float32)


def save_depthmap_xyz(path, depthmap3D):
    """loads depthmap in the original f

    Args:
        save ([str, pathlib.Path]): Path to save the depthmap file.
        depthmap3D (np.ndarray): wxhx3 depthmap float image.

    Raises:
        FileNotFoundError: When depthmap_path does not point to an existing file.

    Returns:
        np.ndarray: depthmap data.
    """

    depthmap_path_p = Path(path).resolve()
    depthmap_path_p.parent.mkdir(parents=True, exist_ok=True)
    suf = depthmap_path_p.suffix
    if suf not in ['.tiff', '.tif']:
        warnings.warn("this functions is designed to store 3 channel depthmaps tiff files. You are tring to store depthmap as a ." + suf
                      + "file")
    depthmap3D[depthmap3D==0]=np.nan#unknown values are set to 0 in parts of code to work with .pngs
    tiff.imsave(str(depthmap_path_p), depthmap3D.astype(np.float32))


ply_header_rgb = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red                   
property uchar green
property uchar blue
end_header
'''


def export_ply(path, scene_points, color_img=None):
    """created an ascii ply file based on scene_points and color_img

    Args:
        path (pathlib.Path, str): path to store the resulting .ply file.
        scene_points (np.ndarray): Nx3 array containing location of points in 3d.
        color_img (np.ndarray, optional): Nx3 array containing bgr color values 
        of each points in scene_points. if not defined, all point will be colored
        white. Defaults to None.
    """
    if color_img is not None:
        assert scene_points.shape == color_img.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = str(path)
    lines = []
    scene_points[scene_points == np.inf] = np.nan
    scene_points[scene_points == -np.inf] = np.nan
    scene_points[scene_points == 0] = np.nan
    
    with open(path, 'w') as ply:
        for i in range(scene_points.shape[0]):
            entry = scene_points[i]
            if np.isnan(entry).any():
                continue
            if color_img is None:
                bgr = [255, 255, 255]
            else:
                bgr = color_img[i]
            ply_entry = [*entry, *bgr[::-1]]
            ply_entry = ' '.join([str(elem)for elem in ply_entry])
            lines.append(ply_entry+'\n')
        ply.write(ply_header_rgb.format(len(lines)))
        for line in lines:
            ply.write(line)
            
def load_ply(path):
    """load only vertexes from ply file.

    Args:
        path (pathlib.Path, str): path to .ply file to load.
        
    Returns:
        nd.array: Nx3 array containing vertexes in loaded ply file.
    """
    path = str(path)
    pts3d_ply = plyfile.PlyData.read(path)
    pts_3d=[]
    for i, elem in enumerate(pts3d_ply['vertex']):
        pts_3d.append(list(elem))
    pts_3d = np.asarray(pts_3d)
    
    return pts_3d


def load_subpix_png(path, scale_factor=256.0):
    """load depthmaps or disparity maps stored as 16bit pngs and normalize values
    using the scale_factor, resulting in float images with subpixel accuracy

    Args:
        path ([pathlib.Path, str]): path of the file to load
        scale_factor (float, optional): the factor used to divide the 16bit 
        integers of the input file. The scaling factor is only been used when the
        pngs are 16bit. Defaults to 256.0.

    Raises:
        FileNotFoundError: when path points to a file that does not exists.

    Returns:
        nd.array: the loaded imagein np.float32 format, normalized if applicable.
        boolean: flag showing if normalization was been done.
    """
    subpix = False
    if not Path(path).is_file():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    disparity = cv2.imread(str(path), -1)
    disparity_type = disparity.dtype
    disparity = disparity.astype(np.float32)
    if disparity_type == np.uint16:
        disparity = disparity / scale_factor
        subpix = True
    return disparity, subpix


def save_subpix_png(path, img, scale_factor=256.0):
    """save a depthmap or a disparity map as png with subpixel accuracy.

    To achieve this, instead of saving information as 8bit image, the function
    multiplies the image by scale_facotr and stores is as a 16bit png. This
    image can be loaded and devided by the same scale_facotr to retreave subpixel
    information. This process is lossy, but allows dephtmaps and disparities to
    be stored as png images for easier previewing.

    Args:
        path ([str, pathlib.Path]): path to store the image.
        img ([type]): the image to store.
        scale_factor (float, optional): the factore that will be used to multiply
        values before storing them as a 16bit integers. Defaults to 256.0.

    Returns:
        [type]: [description]
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = img.astype(np.float32) * scale_factor
    if np.amax(img) > (2**16)-1:
        # warnings.warn("image out of range("+ str(np.amax(img)/scale_factor)+"), try with a smaller scale factor. loading this file will results in invalid values, file: "+str(path),)
        img[img>(2**16)-1]=0
    img = img.astype(np.uint16)
    cv2.imwrite(str(path), img)