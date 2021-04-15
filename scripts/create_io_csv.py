"""Script to create a csv file, containing path for left, right and output_disparites. 
To be used for runing inference on nn.

python -m scripts.create_data_csv ~/path_to_pig_data_root_dir/ ~/disparity_out/method ./csv_save_path.csv
"""

import argparse 
from pathlib import Path
import csv

parser = argparse.ArgumentParser(description='Create csv file containing the paths of left, right and output location to be used with nn inference code.')
parser.add_argument('keyframe_dir', help='root directory of dataset')
parser.add_argument('output_dir', help='directory to store the disparities ')
parser.add_argument('out', help='path to save the dir.')
parser.add_argument('--gt', help='include ground truth reference', action='store_true')
parser.add_argument('--no-header' , help='do not write header', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    root_dir_p = Path(args.keyframe_dir).resolve()
    left_files = [str(f) for f in (root_dir_p/'left_rect').iterdir()]
    right_files = [str(root_dir_p/'right_rect' / Path(f).name) for f in left_files]

    out_files = [str(Path(args.output_dir).resolve()  / (Path(f).name)) for f in left_files]


    if (args.gt):
        disparity_files = [str(root_dir_p/'disparity' / Path(f).name) for f in left_files]
        header = ['left', 'right', 'gt', 'out']
        samples = zip(left_files, right_files, disparity_files, out_files)
    else:
        header = ['left', 'right', 'out']
        samples = zip(left_files, right_files, out_files)


    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not args.no_header:
            writer.writerow(header)
        for sample in samples:
            writer.writerow(sample)