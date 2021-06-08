# SCARED_dataset_toolkit
code to manipulate scared data


# Data convension

all data are loaded and manipulated as 32 bit floats.
in cases where values are not available, we omit them for point clouds and use 
nan for 3D images, depthmaps and disparities. 
To facilitate sample preview, easy we store resulting depthmaps and disparities
as 16bit uint pngs. In order to store subpixel information we scale samples 
in a similar way with kitti.


# scared data format
Keyframes provide pointclouds stored as .obj in ascii mode. Those pointclouds 
were generated from the 3Dimages and contain a nan vertices as the keyframe
3dimages do. When loading those .obj we filter out nan values

Keyframe 3Dimages seem to have been constructed by triangulating pixels between 
left and right image views. We do not know details about the process and whether
or not the the distortion information was used. Unknown pixels are stored as nan
Using the Z channel(depthmap) and the calibration parameters to reconstruct the 
scene does not result to the same information stored in the 3dImages. Possibly
because 3d information was a result of triangulation between views which we 
cannot do since we do not have access to pixel correspondences.


The interpolated ground truth sequences have a different format. Unknown depth
information is stored as zero 3D vectors. Because the ground truth of the keyframe
was used to create all the following frames, the initial pointcloud was transformed
based on the kinematics and projected to the left and right frame of reference

CHECK IF THE USED DISTORTION COEFFICIENTS. 


# Testing

It's really hard to write tests for conversions and use the provided data.
This is because we do not know exactly the method used to create the 3dimages.
Using the provided calibration parameters and depthmaps we cannot recreate the
3d images. This might be due to the data being stored using single precision but
manipulated in double precision or because the 3d images where constructed by
triangulating corresponding pixels obtain from the structured light sequences
(information we do not have) or a combination of both.