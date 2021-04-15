source ~/anaconda3/etc/profile.d/conda.sh
conda activate scareddata

python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_0/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_0/data/scene_points.tar.gz ./ds8_k0.mp4
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_1/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_1/data/scene_points.tar.gz ./ds8_k1.mp4 
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_2/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_2/data/scene_points.tar.gz ./ds8_k2.mp4 
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_3/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset8/keyframe_3/data/scene_points.tar.gz ./ds8_k3.mp4 

python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_0/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_0/data/scene_points.tar.gz ./ds9_k0.mp4
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_1/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_1/data/scene_points.tar.gz ./ds9_k1.mp4 
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_2/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_2/data/scene_points.tar.gz ./ds9_k2.mp4 
python -m scripts.video_depthmap_sequense /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_3/data/rgb.mp4 /home/dimitrisps/Datasets/Scared_stereo/test_data/dataset9/keyframe_3/data/scene_points.tar.gz ./ds9_k3.mp4 