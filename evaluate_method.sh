source ~/anaconda3/etc/profile.d/conda.sh
conda activate scareddata

TESTS_DATA_ROOT_DIR=/home/dimitrisps/Datasets/Scared_stereo/test_data
SAVE_ROOT_DIR=/home/dimitrisps/maxs_dataset/deeppruner_best_new_weights_512_max_disp
save_tiffs_dir=/media/dimitrisps/01D54C9588C3D070/submission_scared/DeepPruner_best

python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_0/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_0/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_0/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_0.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_0
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_1/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_1/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_1/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_1.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_1
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_2/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_2/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_2/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_2.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_2
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_3/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_3/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_3/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_3.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_3
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_4/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_4/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_4/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_4.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_4

python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_0/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_0/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_0/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_0.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_0
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_1/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_1/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_1/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_1.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_1
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_2/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_2/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_2/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_2.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_2
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_3/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_3/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_3/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_3.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_3
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_4/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_4/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_4/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_4.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_4


SAVE_ROOT_DIR=/home/dimitrisps/maxs_dataset/hrs_stage3_512
save_tiffs_dir=/media/dimitrisps/01D54C9588C3D070/submission_scared/HSM_stage3

python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_0/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_0/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_0/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_0.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_0
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_1/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_1/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_1/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_1.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_1
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_2/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_2/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_2/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_2.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_2
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_3/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_3/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_3/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_3.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_3
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_4/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset8/keyframe_4/ ${TESTS_DATA_ROOT_DIR}/dataset8/keyframe_4/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset8/keyframe_4.csv --save_dir ${save_tiffs_dir}/dataset8/keyframe_4

python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_0/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_0/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_0/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_0.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_0
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_1/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_1/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_1/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_1.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_1
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_2/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_2/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_2/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_2.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_2
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_3/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_3/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_3/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_3.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_3
python -m scripts.evaluate_keyframe ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_4/data/ground_truth/ ${SAVE_ROOT_DIR}/dataset9/keyframe_4/ ${TESTS_DATA_ROOT_DIR}/dataset9/keyframe_4/data/stereo_calib.json ${SAVE_ROOT_DIR}/dataset9/keyframe_4.csv --save_dir ${save_tiffs_dir}/dataset9/keyframe_4