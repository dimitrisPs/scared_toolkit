import argparse
from pathlib import Path
import re
import csv
import numpy as np
from tqdm import tqdm
from scareddataset.evaluation import xyz_error, depthmap_coverage
from scareddataset.calibrator import StereoCalibrator
from scareddataset.iotools import load_subpix_png, load_depthmap_xyz, save_depthmap_xyz
from scareddataset.data_maniputlation import disparity_to_original_scared
parser = argparse.ArgumentParser(description='Evaluate disparities over a keyframe')
parser.add_argument('ref_dir', help='path to reference directory containing .tiff files.')
parser.add_argument('comp_dir', help='path to infered disparity directory.')
parser.add_argument('--save_dir', help='path to save .tiff files.')
parser.add_argument('calib', help='path to calibration file.')
parser.add_argument('out', help='directory to store result .csv files')


class running_avg():
    def __init__(self):
        self.sum=0
        self.num=0
    def val(self):
        return self.sum/self.num
    def append(self, val):
        self.sum += val
        self.num += 1
        return self.val()
    def flush(self):
        self.sum=0
        self.num=0
    

if __name__ == "__main__":
    args = parser.parse_args()
    avg = running_avg()
    results = []
    err=[]
    
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    calib = StereoCalibrator().load(args.calib)
    comp_dir = Path(args.comp_dir)
    ref_dir = Path(args.ref_dir)
    regex = r"(\d{6})"
    disparity_paths = sorted([p for p in comp_dir.iterdir()])
    pbar = tqdm(disparity_paths, desc="{}->{} (error):".format(ref_dir.parents[2].name, ref_dir.parents[1].name))
    for disparity_path in pbar:
        
        frame_id = re.findall(regex, disparity_path.stem)[0]
        ref_path = ref_dir / ('scene_points'+frame_id+'.tiff')
        error = [int(float(frame_id))]
        ref = load_depthmap_xyz(ref_path)
        coverage = depthmap_coverage(ref[:,:,0])
        
        disparity, _ = load_subpix_png(disparity_path,128.0)
        comp_scared_depthmap = disparity_to_original_scared(disparity, calib)
        
        #need to save tiff files.
        if args.save_dir is not None:
            save_depthmap_xyz(str(save_dir/('frame'+frame_id+'.tiff')),comp_scared_depthmap[:,:,-1])
        if coverage< 0.1:
            error.extend([np.nan, coverage, 0])
            results.append(error)
            continue
        err = xyz_error(ref, comp_scared_depthmap)
        avg.append(err[0])
        error.extend(err)
        results.append(error)
        pbar.set_description("{}->{} (error): {:5f}".format(ref_dir.parents[2].name, ref_dir.parents[1].name, avg.val()))
        pbar.refresh() # to show immediately the update
    
    results.sort(key=lambda x: x[0])
        
    with open(args.out, 'w') as err:
        csv_writer = csv.writer(err)
        csv_writer.writerows(results)
        
    fid, err, cov, _ = zip(*results)
    err = np.array(err)
    print(np.nanmean(err))

    