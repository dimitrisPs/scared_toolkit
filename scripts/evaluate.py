import sys
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
from pathlib import Path
import scaredtk.io as sio
import pandas as pd

def load_generic_sample(p, scale_factor=256.0):
    if p.suffix=='.tiff':
        sample = tifffile.imread(str(p))
        if sample.shape[-1]==3:
            sample =sample[...,-1]
    else:
        sample = sio.load_subpix_png(p, scale_factor=scale_factor)
    
    return sample



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_test_dir')
    parser.add_argument('root_prediction_dir')
    parser.add_argument('domain', choices=['disparity', 'depth'])
    parser.add_argument('--scale_factor', default=128.0, type=float)
    args = parser.parse_args()


    root_test_dir = Path(args.root_test_dir)
    root_prediction_dir = Path(args.root_prediction_dir)

    # search evaluation index files, that encode the names of frames with > 10%
    # coverage. If you not available, create them
    
    ref_keyframe_dirs = sorted([p for p in root_test_dir.rglob('**/keyframe_*') if p.is_dir()])
    pred_keyframe_dirs = sorted([p for p in root_prediction_dir.rglob('**/keyframe_*') if p.is_dir()])
    eval_lists = sorted([p for p in root_test_dir.rglob('**/valid.csv')])
    if len(eval_lists) != len(ref_keyframe_dirs):
        print('valid frame lists are missing, we need to generate them first')
        for kf in tqdm(ref_keyframe_dirs, desc='keyframes processed'):
            valid_list=[]
            depth_frames = sorted([p for p in (kf/'depthmap').iterdir()])
            if len(depth_frames)==0:
                print('cannot generate valid_list because depthmap directory under keyframe_* does not exist', file=sys.stderr)
                return 1
            for frame_id, depthmap_path in tqdm(enumerate(depth_frames), desc='depthmap processed'):
                
                dmap = load_generic_sample(depthmap_path, args.scale_factor)
                    
                coverage = 1 - (np.count_nonzero(np.isnan(dmap))/dmap.size)
                if coverage>=.1:
                    valid_list.append(frame_id)
            np.savetxt(kf/"valid.csv", np.array(valid_list), delimiter=',')
        eval_lists = sorted([p for p in root_test_dir.rglob('**/valid.csv')])
        
    assert len(eval_lists) == len(ref_keyframe_dirs)
    results_dataset=[]
    results_keyframe=[]
    results_mae=[]
    results_bad3=[]
    for ref_kf, pred_kf in zip(ref_keyframe_dirs, pred_keyframe_dirs):
        
        assert ref_kf.name == pred_kf.name
        assert ref_kf.parent.name == pred_kf.parent.name
        
        valid_ids = list(np.loadtxt(kf/"valid.csv", delimiter=','))
        
        ref_paths = np.array(sorted([p for p in ref_kf/args.domain]))[valid_ids]
        pred_paths = np.array(sorted([p for p in pred_kf/args.domain]))[valid_ids]
        
        
        # load valid indexes
        assert len(ref_paths) == len(pred_paths)
        mae_lst = []
        bad3_lst =[]
        
        for ref_p, pred_p in zip(ref_paths, pred_paths):
            ref = load_generic_sample(ref_p, args.scale_factor)
            pred = load_generic_sample(pred_p, args.scale_factor)
            pred = np.nan_to_num(pred)
            
            error = np.abs(ref-pred)
            mae_lst.append(np.nanmean(error))
            if args.domain == 'disparity':
                bad3_lst.append((np.sum(error>3)/error.size)*100)
                
        assert len(mae_lst) ==len(valid_ids)
        
        results_dataset.append(ref_kf.parent.name[-1])
        results_keyframe.append(ref_kf.name[-1])
        results_mae.append(np.mean(np.array(mae_lst)))
        if args.domain=='disparity':
            results_bad3.append(np.mean(np.array(bad3_lst)))
    
    results_dict = {'dataset':results_dataset, 'keyframe':results_keyframe, 'MAE':results_mae}
    if args.domain=='disparity':
        results_dict['bad3']=results_bad3
    
    results = pd.DataFrame(results_dict)
    
    results.to_csv(root_prediction_dir/f'results_{args.domain}.csv')
    
        
        
if __name__ == "__main__":
    sys.exit(main())