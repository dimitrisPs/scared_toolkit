import errno
import os
import warnings
import numpy as np
import cv2
from pathlib import Path

from plyfile import PlyData, PlyElement
import tifffile as tiff
import tarfile
import re
from collections import OrderedDict
from tqdm import tqdm
from io import BytesIO
from typing import Union, Literal, Tuple, Iterator
import json

ORIENTATION = Literal["vertical", "horizontal"]


def load_pose_sequence(filepath: Union[Path, str]) -> OrderedDict:
    """Loads pose data for the provided ground truth sequences
    
    This function extracts the frame_data.tar.gz files and parses the provided 
    json files for pose data. It completely ignores the calibration data as
    those can be loaded from keyframes. 


    Args:
        filepath (Union[Path, str]): location of the frame_data.tar.gz archive

    Returns:
        OrderedDict: ordered dictionary containing poseses that can be indexed 
        by the frame id
    """
    poses = OrderedDict()
    frame_data_p = Path(filepath)
    with tarfile.open(frame_data_p, "r:gz") as frame_data:
        samples = frame_data.getmembers()
        for sample in tqdm(samples, desc="loading pose data", leave=False):
            with frame_data.extractfile(sample) as sample_json:
                frame_id = int(re.sub(r"\D", "", sample.name))
                pose = np.array(json.loads(sample_json.read())["camera-pose"])
                poses[frame_id] = pose
    return poses


class Img3dTarLoader:
    """ Extract SCARED ground truth .tiff sequence stored in .tar one by one.
    
        This class can be used to load the tarred ground truth sequences 
        containing .tiff files one by one. This is usefull because the sequences
        occupy a lot of scare and might pose a problem in systems with limited
        hard drive space. 
    
    """

    def __init__(self, tar_filepath: Union[Path, str]) -> None:
        """Constructor

        Args:
            tar_filepath (Union[Path, str]): path to read the tarred .tiff
            sequence from.

        Returns:
            None: [description]
        """
        tar_p = Path(tar_filepath)
        assert tar_p.is_file()
        self.tardata = tarfile.open(tar_p, "r:gz")
        # contract a reference dict to access .tiff files based on their index
        self.tarnames = {
            int(re.sub(r"\D", "", i.name)): i for i in self.tardata.getmembers()
        }
        self.num_frames = len(self.tarnames)

    def __getitem__(self, key: int) -> np.ndarray:
        assert isinstance(key, int)
        assert key < self.num_frames
        data = self.tardata.extractfile(self.tarnames[key]).read()
        img = tiff.imread(BytesIO(data))
        # items with z=0 are clearly unknown, set them to nan.
        img[img[:, :, 2] == 0] = np.nan
        return img

    def __iter__(self) -> Iterator[np.ndarray]:
        self.idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.idx < self.num_frames:
            x = self.idx
            self.idx += 1
            return self.__getitem__(x)
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.num_frames

    def __exit__(self) -> None:
        self.tardata.close()
        del self.tardata
        del self.tarnames
        del self.num_frames


class stereo_video_capture(cv2.VideoCapture):
    """ Wraps cv2.VideoCapture to read stereo videos stored as stacked images"""

    def __init__(
        self, videopath: Union[Path, str], stacked: ORIENTATION = "vertical"
    ) -> None:
        """Constructor
        
        Args:
            videopath (Union[Path, str]): path to load the stereo video from
            stacked (ORIENTATION, optional): whether the left and right channels
            are stacked vertical or horizontal to each other. The left channel is
            always assumed to be first(on top or on the left for vertical and
            horizontal stacking respectively). Defaults to "vertical".

        Returns:
            None: 
        """
        super().__init__(str(videopath))
        assert stacked in ["vertical", "horizontal"]
        self.stacked = stacked
        assert super().isOpened()
        self.w = int(super().get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(super().get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """reads the next frame 
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple containing the left and the 
            right stereo frame
        """
        ret, frame = super().read()
        if ret is None:
            return None, None
        if self.stacked == "vertical":
            left = frame[: self.h // 2]
            right = frame[self.h // 2 :]
        else:
            left = frame[:, : self.w // 2]
            right = frame[:, self.w // 2 :]
        return left, right


def load_ply_as_ptcloud(path: Union[Path, str]) -> np.ndarray:
    """Loads a pointcloud stored as ply
    
    The function does not support RGB infomation and does not clean data like
    the load_scared_obj does.

    Args:
        path (Union[Path, str]): path to load the .ply from

    Returns:
        np.ndarray: Nx3 array, containing the 3D coordinated of N points.
    """
    with open(path, "rb") as f:
        data = PlyData.read(f)
        print((data["vertex"]["x"]).shape)
        ptcloud = np.vstack(
            [data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]
        ).T
    return ptcloud


def load_scared_obj(path: Union[Path, str]) -> np.ndarray:
    """loads an ascii .obj pointcloud
    
    Loads a .obj pointcloud like the those provided in the SCARED keyframes.
    The scared provided .obj files contain points with nan coordinates. When 
    loading those files, this function removes all nan entries and keeps only
    valid points. 

    Args:
        path (Union[Path, str]): path to load the .obj file from

    Returns:
        np.ndarray: Nx3 array, containing the 3D coordinates of N points.
    """
    pts = []
    with open(path, "r") as f:
        objdata = f.readlines()
    for entry in objdata:
        if "nan" in entry:
            continue
        entry = entry.strip().split(" ")
        if entry[0] != "v":
            continue
        pts.append([float(n) for n in entry[1:]])
    ptcloud = np.array(pts, dtype=np.float32)
    return ptcloud


def load_img3d(path: Union[Path, str]) -> np.ndarray:
    """loads a 3 channel .tiff image as a 3D image
    
    This functions loads a 3 channel .tiff as a 3D Image which is the format
    used to store ground truth in SCARED. Each channel of this image corresponds
    to a cartesian coordinate and the pixel location encode the projection 
    location of the encoded 3D point in the camera frame. Because ground truth
    .tiff file format is not always consistent, in order to standarise data
    format for this project we choose to express points located in the origin 
    with nan values. This is the convention used in the keyframes but not in
    the sequence. The function ensured consistency when loading a gt sample, 
    regardless if it comes from the interpolated sequence or a keyframe. 

    Args:
        path (Union[Path, str]): path to load the img3d file from

    Raises:
        FileNotFoundError: when path argument points to a non existent file.

    Returns:
        np.ndarray: the loaded 3d image array in single float precision.
    """

    img3d_p = Path(path).resolve()
    if not img3d_p.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    img3d = tiff.imread(str(img3d_p))
    img3d[img3d[:, :, 2] == 0] = np.nan  # set unknown points as nan

    return img3d.astype(np.float32)


def load_subpix_png(path, scale_factor=256.0) -> np.ndarray:
    """load float array stored as a png img
    
    This function load float arrays stored as 16bit uint .png images and scales
    them down by scale_factor to restore decimal infomation. If path points to 
    a 8bit png, scale factor is ignored decimal infomation is assumed
    absent.Because save_subpix_png replace nan and inf values with zeros while
    saving them as .png, this functions replaces zeros values with nan to
    indicate that infomation for those pixels is unknown.

    Args:
        path (Union[Path, str]): path to load the .png file from
        scale_factor (float, optional): the factor by which the values of img
        are scaled in order to span the 16-bit uint range. The same value must
        be used when loading float arrays stored by this function. Defaults to
        256.0


    Raises:
        FileNotFoundError: Path points to a file that does not exist

    Returns:
        np.ndarray: the loaded float array in single presision.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    img = cv2.imread(str(path), -1)
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.uint16:
        img = img / scale_factor
    img[img == 0] = np.nan
    return img


def save_img3d(path: Union[Path, str], img3d: np.ndarray) -> Path:
    """Saves a float WxHx3 array as .tiff file.
    
    This function is used to store pointcloud that encode projection pixel 
    information as .tiff files, the same format used to store ground truth
    infomation in SCARED. Each channel correspond to a dimention in the 
    cartesian space and the u,v location of a 3D point encode the place where
    it projects to.

    Args:
        path (Union[Path, str]): path to store the resulting .tiff
        img3d (np.ndarray): WxHx3 pointcloud map to store

    Returns:
        Path: the input path argument as a pathlib.Path 
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    assert img3d.dtype == np.float32

    suf = path.suffix
    if suf not in [".tiff", ".tif"]:
        warnings.warn(
            "this functions is designed to store 3 channel depthmaps tiff files. You are tring to store depthmap as a ."
            + suf
            + "file"
        )
    tiff.imsave(str(path), img3d.copy().astype(np.float32))


def save_subpix_png(
    path: Union[Path, str], img: np.ndarray, scale_factor: float = 256.0
) -> Path:
    """Save a float 1 channel array as a png, encoding subpixel information
    
    This function is used to save one channel float arrays, e.g. depthmaps and
    disparity maps, as png images to facilitate previewing, while encoding 
    decimal information. The float arrays are stored as 16bit uint .png and 
    decimal infomation is encoded by scaling img by scale_factor to span the 
    whole 16bits. After casting the scaled img to uint, there is still loss of
    information.  Infinate and nan values are replaced by 0.

    Args:
        path (Union[Path, str]): path to store the resulting .png file
        img (np.ndarray): WxH input float image
        scale_factor (float, optional): the factor by which the values of img
        are scaled in order to span the 16-bit uint range. The same value must
        be used when loading float arrays stored by this function. Defaults to
        256.0.

    Returns:
        Path: the input path argument as a pathlib.Path 
    """

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    img = img.copy() * scale_factor

    if np.amax(img) > (2 ** 16) - 1:
        warnings.warn(
            "image out of range("
            + str(np.amax(img) / scale_factor)
            + "), try with a smaller scale factor. loading this file will results in invalid values, file: "
            + str(path),
        )
        img[img > (2 ** 16) - 1] = 0
    img = np.nan_to_num(img, posinf=0, neginf=0)
    cv2.imwrite(str(path), img.astype(np.uint16))
    return path


def save_ptcloud_as_ply(
    path: Union[Path, str], ptcloud: np.array, save_binary: bool = True
) -> Path:
    """save an Nx3 array as a .ply file
    
    This functions accept a N element pointcloud stored as an Nx3 np.array
    and stores is at as .ply. The function can store pointcloud both in 
    ascii and binary mode, according to the save_binary flag. Additionally
    if the destination folder does not exist, the function creates it and stores
    the resulting pointcloud. Currently the function does not support storing 
    of RGB information.

    Args:
        path (Union[Path, str]): path to store ptcloud as a .ply file
        ptcloud (np.array): N element pointcloud stored as a Nx3 np.array
        save_binary (bool, optional): Save .ply in binary mode. Defaults to True.
        
    Returns:
        Path: the input path argument as a pathlib.Path 
    """
    assert ptcloud.dtype == np.float32

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.array(
        list(zip(ptcloud[:, 0], ptcloud[:, 1], ptcloud[:, 2])),
        dtype=[("x", "float32"), ("y", "float32"), ("z", "float32")],
    )
    el = PlyElement.describe(vertices, "vertex")

    if save_binary:
        PlyData([el]).write(str(path))
    else:
        PlyData([el], text=True).write(str(path))
    return path

