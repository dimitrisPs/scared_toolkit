# SCARED dataset toolkit


This repository contains code to help you generate additional data, such as
disparity and depthmap samples. The repository also provides tools to create the
interpolated ground truth sequences based on keyframe ground truth data.


## Data convension

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

### 3D Images (.tiff)
The provided .tiff keyframe files encode unknown values as nan, whereas the 
interpolated .tiff files in the sequences zeros(we haven't check every sequence).
Since we want all unknown points to have nan values, when loading, our functions
replace 0 vectors with nan values.

### Disaprity and depthmaps (.png)

To facilitate sample preview we store both generated disparity and depthmaps
as 16bit uint pngs. All depth values are in mm distance and disparity is measured
as the difference in y directions between the coordinates of a point in the left
image with its corresponding point in the right image. In order to maintain
decimal information when storing samples as .png, we scale the disparity and
depth values by a configurable argument called scale_factor(default is 256.0)
this maps a range of 0-255 to a range of 0-65280 and then store them as 16bit
unsigned integers. If there is a need of storing values greater than 255, one 
can adjust the the scale factor to something that will cast the sample range
to span values 0-2^16. Nan values are stored as 0. When loading such samples the 
scale_factor is used to remap 0-2^16 values to the correct range and 0 values 
are replaced by nan. This process is obviously lossy but it maintains correct
information up to 2 decimal points when a scale_factor of 256.0 is used.


## Provided function

The repository, provides python code to load and save samples provided with
the original dataset as well as functions to store and load depthmap and
disparity samples with decimal information encoded in 16-bit uint .png.

In addition we provide code to manipulate samples and create additional data,
such as disparity samples. Included functions are able to generate depthmaps, 
3D images, disparities and pointclouds from any of the aforementioned domains.
Still if you are to use the the provided functions and not the scripts you need
to check the validity of the outcome. For instace, `ptcloud_to_disparity()` can 
generate a disparity image based on a pointcloud, the result is meaningles if 
the provided pointcloud is not rotated to the rectified frame of reference and
the Projection matrices are not obtained from the stereo rectification process.


## Usage

### Generate keyframe

### Extract ground truth sequence

### Generate ground truth sequence




## Testing

It's really hard to write tests for conversions and use the provided data.
This is because we do not know exactly the method used to create the 3dimages.
Using the provided calibration parameters and depthmaps we cannot recreate the
3d images. This might be due to the data being stored using single precision but
manipulated in double precision or because the 3d images where constructed by
triangulating corresponding pixels obtain from the structured light sequences
(information we do not have) or a combination of both.