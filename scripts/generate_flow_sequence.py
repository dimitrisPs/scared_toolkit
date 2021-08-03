from pathlib import Path
from scaredtk.calibrator import StereoCalibrator
import scaredtk.io as sio
from scaredtk.convertions import ptcloud_to_flow
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import tifffile as tiff

parser =argparse.ArgumentParser(description="Create keyframe dataset")
parser.add_argument('root_dir', help="root directory under which keyframe data are stored")
parser.add_argument('--recursive', '-r', help='scans for keyframe_* directories under root_dir and processes them all', action='store_true')
parser.add_argument('--out_dir', help='where to store the resulting dataset, if not set, generated files will be stored in src folders')
parser.add_argument('--ptcloud', help='name of the pointcloud to provide reference, .ply are supported, must be places inside keyframe dirs.')









if __name__=='__main__':
    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    #recursively find all keyframe dirs
    if args.recursive:
        keyframe_dirs = [p for p in root_dir.rglob('**/keyframe_*') if p.is_dir()] 
    else:
        keyframe_dirs = [root_dir]  
    for kf in tqdm(keyframe_dirs,desc='processed keyframes'):
        out_dir = Path(args.out_dir)/kf.parent.name/kf.name if args.out_dir is not None else kf
        out_dir.mkdir(exist_ok=True, parents=True)
        
        stereo_calib = StereoCalibrator()
        calib = stereo_calib.load(kf/'endoscope_calibration.yaml')
        
        pose_dict = sio.load_pose_sequence(kf/'data'/'frame_data.tar.gz')
        
        if args.ptcloud is not None:
            gt_ptcloud = sio.load_ply_as_ptcloud(kf/args.ptcloud)
        else:
            gt_ptcloud = sio.load_scared_obj(kf/'point_cloud.obj')
        
        sample_sequence = [(fid, pose_dict[fid]) for fid in pose_dict.keys()]
        
        previous_pose=None
        for fid, pose in tqdm(sample_sequence, desc='sample', leave=False):
            
            if previous_pose is None:
                previous_pose = pose
                continue
            
            forward_flow = ptcloud_to_flow(gt_ptcloud, previous_pose, pose, [1024,1280], calib['K1'], calib['D1'])
            
            sio.save_flow_kitti(str(out_dir/'forward_flow'/f'{fid-1:06d}.png'), forward_flow)
            previous_pose = pose
                    