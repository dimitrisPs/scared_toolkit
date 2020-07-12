import argparse
from pathlib import Path
import re
import csv
import numpy as np
from tqdm import tqdm
from scareddataset.evaluation import xyz_error
from scareddataset.calibrator import StereoCalibrator
from scareddataset.iotools import load_subpix_png, load_depthmap_xyz
from scareddataset.data_maniputlation import disparity_to_original_scared
parser = argparse.ArgumentParser(description='Evaluate disparities over a keyframe')
parser.add_argument('ref_dir', help='path to reference directory containing .tiff files.')
parser.add_argument('comp_dir', help='path to infered disparity directory.')
parser.add_argument('calib', help='path to calibration file.')
parser.add_argument('out', help='directory to store result .csv files')


if __name__ == "__main__":
    args = parser.parse_args()
    
    results = []
    calib = StereoCalibrator().load(args.calib)
    comp_dir = Path(args.comp_dir)
    ref_dir = Path(args.ref_dir)
    regex = r"(\d{6})"
    disparity_paths = [p for p in comp_dir.iterdir()]
    for disparity_path in tqdm(disparity_paths, desc=ref_dir.parents[1].stem+': '):
        frame_id = re.findall(regex, disparity_path.stem)[0]
        ref_path = ref_dir / ('scene_points'+frame_id+'.tiff')
        
        
        disparity, _ = load_subpix_png(disparity_path)
        comp_scared_depthmap = disparity_to_original_scared(disparity, calib)
        
        ref = load_depthmap_xyz(ref_path)
        error = [int(float(frame_id))]
        error.extend(xyz_error(ref, comp_scared_depthmap))
        results.append(error)
    
    results.sort(key=lambda x: x[0])
        
    with open(args.out, 'w') as err:
        csv_writer = csv.writer(err)
        csv_writer.writerows(results)
        
    fid, err, cov, _ = zip(*results)
    err = np.array(err)
    print(np.nanmean(err))

    