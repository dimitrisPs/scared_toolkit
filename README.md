# SCARED dataset toolkit

This repository contains Unofficial code to help you generate additional data,
such as disparity, depthmap and flow samples in the left rectified frame of reference.
Additionally tools are provided to generate the interpolated ground truth
sequences based on keyframe ground truth data.

If you are using the [Stereo correspondence and reconstruction of endoscopic data](https://endovissub2019-scared.grand-challenge.org/Home/)
cite the [challenge's paper](https://arxiv.org/abs/2101.01133)

```cite
@article{allan2021stereo,
  title={Stereo correspondence and reconstruction of endoscopic data challenge},
  author={Allan, Max and Mcleod, Jonathan and Wang, Congcong and Rosenthal, Jean Claude and Hu, Zhenglei and Gard, Niklas and Eisert, Peter and Fu, Ke Xue and Zeffiro, Trevor and Xia, Wenyao and others},
  journal={arXiv preprint arXiv:2101.01133},
  year={2021}
}
```

This repository contains code developed as a part of the paper [Real-time multi-task 3D scene reconstruction and instrument segmentation using surgical video]().
If you end up using code provided in this repository, please consider citing

```cite

```

## Getting access to the SCARED dataset

The dataset was made publicly available after the completion of the SCARED challenge.
To get access you need to go to the [SCARED challenge download webpage](https://endovissub2019-scared.grand-challenge.org/Downloads/),
create a user account in the grand-challenge platform, join the challenge and follow the provided instructions.

## Data convention

We've established a data format to facilitate development. Depth and disparity
data are loaded and manipulated as floats. If disparity or depth information is
not available for a specific pixel, its values are represented by nan.

### Pointclouds

The .obj pointclouds, provided with every keyframe, contain HxW points,
with H, W the height and with of the monocular frame. Since SCARED does not
provide full coverage, some of those vertices are represented as nan. Our
loading functions remove such points which results to much smaller pointclouds
containing only points for which we know ground truth information. Although
not used by our scripts, we provide code to save pointclouds as ply. In that
case we save only points with known ground truth.

### 3D Images (pointmaps) (.tiff)

The provided .tiff keyframe files encode unknown values as nan, whereas the provided
interpolated .tiff files in the sequences zeros(we haven't check every sequence).
Since we want all unknown points to have nan values, when loading, our functions
replace 0 vectors with nan values.

### Disaprity and depthmaps (.png)

To facilitate sample preview, we store both generated disparity and depthmaps
as 16bit uint pngs. All depth values are in mm distance and disparity is measured
as the difference in y directions between the coordinates of a point in the left
stereo rectified image with its corresponding point in the right stereo rectified image. In order to maintain
decimal information when storing samples as .png, we scale the disparity and
depth values by a configurable argument called scale_factor(default is 256.0)
This maps a range of 0-255 to 0-65280 and then store them as 16bit
unsigned integers. If there is a need of storing values greater than 255, one
can adjust the the scale factor to something that will cast the sample range
to span values 0-2^16. Nan values are stored as 0. When loading such samples the
scale_factor is used to remap 0-2^16 values to the correct range and 0 values
are replaced by nan. This process is obviously lossy but it maintains correct
information up to 2 decimal points when a scale_factor of 256.0 is used.

## Usage

In addtition to hight level data extraction and disparity generation scripts
the repository, provides python code to load and save samples provided with
the original dataset as well as functions to store and load depthmap and
disparity samples with decimal information encoded in 16-bit uint .png.
It also includes code to manipulate samples and create additional data,
such as disparity samples. Included functions are able to generate depthmaps,
3D images, disparities and pointclouds from any of the aforementioned domains.
Still if you are to use the the provided functions and not the scripts you need
to check the validity of the outcome. For instace, `ptcloud_to_disparity()` can
generate a disparity image based on a pointcloud, the result is meaningless if
the provided pointcloud is not rotated to the rectified frame of reference and
the Projection matrices are not obtained from the stereo rectification process.


### Environment setup

This project was build using anaconda. Assuming that anaconda is already installed
in the target machine, a anaconda environment suitable to run this code can be
created using the following steps.

- navigate to this project's folder
- create an environments (e.g. scared_toolkit) using the provided requirements.txt

```bash
conda create --name scared_toolkit --file requirements.txt
```

activate the anaconda environment

```bash
conda activate scared_toolkit
```

### Initial file structure

All the scripts listed bellow expect an standard file structure. Before using
ensure that the initial scared dataset follows the file structure described bellow

```tree
.
├── dataset_1                           # each subdataset folder should follow the dataset_{$dataset_number} notation
│   ├── keyframe_1                      # each keyframe folder should follow the keyframe_{$keyframe_number} notation
│   │   │    endoscope_calibration.yaml
│   │   │    left_depth_map.tiff
│   │   │    Left_Image.png
│   │   │    point_cloud.obj
│   │   │    right_depth_map.tiff
│   │   │    Right_Image.png
│   │   └── data
│   │       ├── frame_data.tar.gz
│   │       ├── rgb.mp4
│   │       └── scene_points.tar.gz
:   :       :
│   └── keyframe_M
│       └── data
:       :
└── dataset_N
    ├── keyframe_1
    :       :         
    └── keyframe_M
```

### Extract ground truth sequence plus disparity, rectified views etc

This script does the following:

- splits the rgb.mp4 into left and right .png images
- unpacks and the contents of the scene_points.tar.gz and splits them to left and right
- reads the endoscope calibtration.yaml, port it to opencv format and add stereo rectification related parameters
- [optional] undistort the rgb frames and depthmaps `--undistort`
- [optional] generate depthmap in .png format `--depth`
- [optional] stereo rectify the rgb images from the rgb.mp4 `--disparity`
- [optional] stereo rectify depthmaps  `--disparity`
- [optional] generate left disparity maps `--disparity`

```bash
python -m scripts.extract_sequence_dataset root_dir [--out_dir] [--recursive] [--depth] [--undistort] [--disparity] [--alpha] [--scale_factor]
```

`root_dir` root directory under which keyframe data are stored
`--out_dir` where to store the resulting dataset, if not set, generated files will be stored in src folders
`--recursive` scans for keyframe_* directories under root_dir and processes them all

`--depth`generate_depthmap in the original frame of reference (.pngs)
`--undistort`generate undistorted depthmap and left rgb in the original frame of reference
`--disparity`generate rectified views and disparity maps
`--alpha` corresponds to the alpha rectification parameters used in the OpenCV stereo rectification function
`--scale_factor` refer to [this](#disaprity-and-depthmaps-(.png))

### Generate keyframe only dataset plus disparity, rectified views etc

This scripts offers the same functionality as extract_sequence_dataset but with
the exception that it generates a smaller dataset only using the keyframes, completely
ignoring the rgb.mp4 and scene_points.tat.gz sequences. Additionally it offers the
ability to overwrite the provided .obj groundtruth pointcloud.

```bash
python -m scripts.generate_keyframe_dataset root_dir [--out_dir] [--recursive] [--depth] [--undistort] [--disparity] [--pt_cloud] [--alpha] [--scale_factor]
```

`--ptcloud` name of the pointcloud to provide reference, .ply are supported, must be placed inside keyframe dirs.

### Generate Dataset based on only on keyframes plus disparity, rectified views etc

This scripts offers the same functionality as extract_sequence_dataset but with
the exception that it generates the ground truth interpolated sequence
based on the point_cloud.obj( can be overwritten with a .ply file) and the frame_data.tar.gz
endoscope pose sequence. This has the advantage of completely removing the overall size of the
dataset the scene_points.tat.gz files from the dataset reducing its size to only 7.2GB
and making it portable. The generated files are not the exact same  as the contents of
the scene_points.gz files possibly due to numerical precision. For this reason the use
of this script should be limited to only the training datasets and not the test datasets.

```bash
python -m scripts.generate_sequence_dataset root_dir [--out_dir] [--recursive] [--depth] [--undistort] [--disparity] [--pt_cloud] [--alpha] [--scale_factor]
```

### Generate flow sequence

This scripts can be used to generate flow maps and store them in the same format
used by kitti. The command line interface is similar to the previous scripts
but only supports flow generation in the original frame of reference.

```bash
python -m scripts.generate_flow_sequence root_dir [--recursive] [--out_dir] [--ptcloud]
```
