import cv2
import numpy as np
from tmi_dataset.data_processing import read_calib_file, ply_header
import argparse
import pandas as pd
from tqdm import tqdm 
from pathlib import Path


def get_sample_paths(sample_csv, disp_dir):
    samples = pd.read_csv(sample_csv)
    left_rect_img_paths = samples['IMG_REC_left']
    left_rect_mask = samples['MASK_REC_eval']
    calib_file = samples['Calibration']
    disp_paths = []

    for sample in left_rect_img_paths:
        sample_p = Path(sample)
        disp_dir_p = Path(disp_dir) /Path(sample_p.parents[1].name)
        disp_name = sample_p.parents[0].name + '_result.png'
        results_subdata_dir = disp_dir_p / Path(disp_name)
        disp_paths.append(str(results_subdata_dir))
    return disp_paths, calib_file, left_rect_mask, left_rect_img_paths


def compute_projection_mats(calib_path):

    with open(calib_path, "r") as calib_d:
        param_left, param_right, H1, H2 = calib_parser(calib_d)
    K1 = param_left['M']
    K2 = param_right['M']


    H1 = np.linalg.inv(H1)
    H2 = np.linalg.inv(H2)

    # Compute rotation between undistorted and rectified frames R = inv(K) * H * K
    R1 = np.linalg.inv(K1).dot(H1.dot(K1))
    R2 = np.linalg.inv(K2).dot(H2.dot(K2))

    RTL1 = np.eye(4)

    RTR1 = np.eye(4)
    RTR1[:3, :3] = param_right['R']
    RTR1[:3, 3] = param_right['T']

    RTL2 = np.eye(4)
    RTL2[:3, :3] = R1

    RTR2 = np.eye(4)
    RTR2[:3, :3] = R2

    # Compute final transformation matrices -> final = second * first
    RTL = RTL2.dot(RTL1)
    RTR = RTR2.dot(RTR1)

    # Compute projection matrices P = K * [R|T]
    P1 = K1.dot(RTL[:3])
    P2 = K2.dot(RTR[:3])
    return P1, P2


def calib_parser(calib_data):
    data = calib_data.readlines()
    # first camera parameters idx 3
    param_left = camera_param_parser(data[3])

    param_right = camera_param_parser(data[5])
    
    homography_left = values_to_array(data[9].split(), (3, 3),_dtype=np.float)
    homography_right = values_to_array(data[10].split(), (3, 3),_dtype=np.float)
    return param_left, param_right, homography_left, homography_right


def camera_param_parser(camera_calib_data):
    values = camera_calib_data.split()
    size = (int(float(values[0])), int(float(values[1])))
    camera_mat = values_to_array(values[2:11], (3, 3))
    dist_coeffs = values_to_array(values[11:19], (1, 8))
    R = values_to_array(values[19:28], (3, 3),_dtype=np.float)
    T = values_to_array(values[28:31], (1, 3),_dtype=np.float)
    return {'size': size, 'M': camera_mat, 'D': dist_coeffs, 'R': R, 'T': T}

def values_to_array(values, out_size=(-1), _dtype=np.float32):
    output = [float(elem) for elem in values]
    output = np.asarray(output, dtype=_dtype)
    output = output.reshape(out_size)
    return output
def create_parent_dirs(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True) 
            
def export_ply(path, scene_points, color_img=None):
    # creates a .ply file containing scene_points 3D locations
    path = str(path)
    create_parent_dirs(path)
    lines =[]
    with open(path, 'w') as ply:
        for i in range(scene_points.shape[0]):
            entry = scene_points[i]
            if (entry[0] == -np.inf or entry[0] == np.inf):
                entry = [0, 0, 0]
                continue
            if color_img is None:
                bgr = [255,255,255]
            else:
                bgr = color_img[i]
            ply_entry = [*entry, *bgr[::-1]]
            ply_entry = ' '.join([str(elem)for elem in ply_entry])
            lines.append(ply_entry+'\n')
        ply.write(ply_header.format(len(lines)))
        for line in lines:
            ply.write(line)
            


def export_xyz(scene_points, file_path):
    with open(file_path, 'w') as f:
        idxs = np.nonzero(scene_points)
        known_depths = list(zip(*idxs[:2]))
        for depth_idx in known_depths:
            xyz = scene_points[depth_idx]
            xyz_entry = [depth_idx[1], depth_idx[0],  *xyz]
            entry = ' '.join([str(elem)for elem in xyz_entry])
            f.write(entry+'\n')

def disparity_to_3d(disparity, P1, P2, mask = None):

    height, width = disparity.shape[:2]
    if mask is not None:

        mask = (255-mask)/255
        disparity = disparity * mask

    idxs = np.nonzero(disparity)
    known_disps = list(zip(*idxs))

    scene_points = np.zeros((height, width, 3), dtype=np.float)

    for disp_idx in known_disps:
        disp = disparity[disp_idx]
        left = np.asarray([disp_idx[1], disp_idx[0]], dtype=np.float)
        right = np.asarray([disp_idx[1] - disp, disp_idx[0]], dtype=np.float)

        point_3d_h = cv2.triangulatePoints(P1, P2, left, right)
        point_3d = (point_3d_h[:3]/point_3d_h[3]).squeeze()

        scene_points[disp_idx] = point_3d

    return scene_points


def main():

    parser = argparse.ArgumentParser(
        description='Disparity image to 3D scene points.')
    parser.add_argument(
        '--csv', help='Path to csv file containing sample file locations.')
    parser.add_argument('--disp_dir', help='Path containing the disparities')
    args = parser.parse_args()

    disp_p, calib_p, rect_mask_p, left_rect_img_p = get_sample_paths(
        args.csv, args.disp_dir)
    for i in tqdm(range(len(disp_p))):

        disp = cv2.imread(disp_p[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(rect_mask_p[i], cv2.IMREAD_GRAYSCALE)
        bgr = cv2.imread(left_rect_img_p[i])


        P1, P2 = compute_projection_mats(calib_p[i])


        scene_pts = disparity_to_3d(disp, P1, P2)

        xyz_path = Path(disp_p[i]).parent / Path(str(Path(disp_p[i]).stem) +'.xyz')
        # ply_path = Path(disp_p[i]).parent / Path(xyz_path.stem + '.ply')



        export_xyz(scene_pts, xyz_path)
        # export_ply(scene_pts, bgr, ply_path)



if __name__ =='__main__':
    main()
