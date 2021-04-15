import argparse
import cv2
import numpy as np
import tarfile
from pathlib import Path
import shutil
from scareddataset.iotools import load_depthmap_xyz
from scareddataset.data_maniputlation import scared_to_depthmap

parser = argparse.ArgumentParser(description='create left rgb, depth image video')

parser.add_argument('input_video', help='scared video sequence.')
parser.add_argument('scenepoints_tar', help='tar file containing scenepoints sequence.')
parser.add_argument('out_video', help='path to save the resulting video file.')



if __name__ == "__main__":
    args = parser.parse_args()
    
    
    tar_path = Path(args.scenepoints_tar)
    in_video_path = Path(args.input_video)
    
    if not tar_path.is_file():
        print('check scenepoints_tar path')
        exit()
    extract_dir = tar_path.parent / 'extruct_dir'
    extract_dir.mkdir(exist_ok=True, parents=True)
    
    if not in_video_path.is_file():
        print('check path to input video.')
        exit()
    
    # create a temp file and extruct scenepoints from tar.
    tar = tarfile.open(str(tar_path), "r:gz")
    tar.extractall(str(extract_dir))
    tar.close()
    
    scene_point_paths = sorted([str(path) for path in extract_dir.iterdir()])
       
    
    invideo = cv2.VideoCapture(args.input_video)
    Path(args.out_video).parent.mkdir(exist_ok=True, parents=True)
    outvideo = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280//2,1024//4))
    i=0
    
    delay=0
    while (invideo.isOpened()):
        # Capture frame-by-frame
        ret, rgb = invideo.read()
        if not ret:
            break
        
        scared_gt = load_depthmap_xyz(scene_point_paths[i])
        
        depthmap_mono = scared_to_depthmap(scared_gt)
        depthmap = np.stack([depthmap_mono]*3, axis =2).astype(np.uint8)

        out_frame = rgb.copy()
        # out_frame[depthmap.nonzero()] = depthmap[depthmap.nonzero()]
        fout = np.zeros((1024,1280*2,3))
        fout[:,:1280]=out_frame[:1024,:]
        fout[:,1280:]=out_frame[1024:,:]
        
        fout = cv2.resize(fout, (1280//2,1024//4), interpolation = cv2.INTER_AREA)
        

        outvideo.write(fout.astype(np.uint8))
        i+=1
    shutil.rmtree(str(extract_dir))
    invideo.release()
    outvideo.release()
