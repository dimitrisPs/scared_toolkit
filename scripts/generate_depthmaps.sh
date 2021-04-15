ROOT_DATASET_DIR=/home/dimitrisps/Datasets/Scared_stereo/scared
DST_DIR=/home/dimitrisps/Datasets/Scared_stereo/train_keyframes

# mkdir $DST_DIR


for dataset in $ROOT_DATASET_DIR/*; do
    dataset_name=$(basename $dataset)
    testss=`($dataset | grep -Eo '[0-9]+$')`

    for keyframe in $dataset/*; do
        # keyframe
        
        # dataset_number=$(${dataset}|grep -Eo '[0-9]+$')
        # keyframe_number="${keyframe}|grep -Eo '[0-9]+$'"
        # num=`"${dataset}" | grep -Eo '[0-9]+'`

        echo $dataset_name
        # echo ${keyframe} ${dataset_number}_${keyframe_number}.png
        exit
    done

done