#!/usr/bin/env python3

import argparse
import sys
import shutil
from pathlib import Path
from tqdm import tqdm 



def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument('src_dataset', help='source root directory keyframe only dataset is stored.')
    parser.add_argument('dst_dataset', help='destination root direcotry to store the aggregated dataset.')
    parser.add_argument('--overwrite', help=' overwrite previous data', action='store_true')
    args = parser.parse_args()
    
    # find all keyframes directoreis under src_dataset directory.
    src_dataset = Path(args.src_dataset)
    dst_dir=Path(args.dst_dataset)
    keyframe_dirs = [p for p in src_dataset.rglob('**/keyframe_[0-5]') if p.is_dir()]
    if not keyframe_dirs:
        print('check src_dataset path, no keyframe folders found', file=sys.stderr)
        return 1
    # create dst path
    for keyframe_dir in tqdm(keyframe_dirs, desc='keyframes processed1'):
        file_lst = [fp for fp in keyframe_dir.iterdir() if fp.is_file()]
        
        for fp in file_lst:
            fp_dst_dir = dst_dir/fp.stem
            fp_dst_dir.mkdir(parents=True, exist_ok=True)
            dst_fp = fp_dst_dir/(f'{keyframe_dir.parent.name[-1]}_{keyframe_dir.name[-1]}{fp.suffix}')
            if not dst_fp.exists() or args.overwrite:
                shutil.copy(fp, dst_fp)
            else:
                print("dst files already exist, call the script with the overwrite flag.", file=sys.stderr)
                return 1
    return 0
    

if __name__ == "__main__":
    sys.exit(main())