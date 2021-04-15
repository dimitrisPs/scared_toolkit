test_root_dir=/home/dimitrisps/Datasets/Scared_stereo/test_data
results_dir=/home/dimitrisps/maxs_dataset/hrs_stage3

for dataset in $test_root_dir/*;do
    for keyframe in $dataset/*;do
        data_dir=$keyframe/data
        [ ! -d "$data_dir" ] && continue
        gt_dir=$data_dir/ground_truth
        comp_dir=$results_dir/$(basename $dataset)/$(basename $keyframe)
        [ ! -d "$comp_dir" ] && continue
        calib=$data_dir/stereo_calib.json
        save_results=$results_dir/$(basename $dataset)_$(basename $keyframe).csv
        python -m scripts.evaluate_keyframe $gt_dir $comp_dir $calib $save_results
    done

done