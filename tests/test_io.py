# from scareddataset.scaredframe import ScaredFrame
import numpy as np
from pathlib import Path

from scaredtk import io as sio


def test_save_tiff():
    target_file = Path('test.tiff')
    ref_file = sio.load_img3d('./data/keyframe_1/left_depth_map.tiff')
    sio.save_img3d(target_file, ref_file)
    reloaded_file = sio.load_img3d(target_file)
    np.testing.assert_equal(ref_file, reloaded_file)
    assert np.nanmean(ref_file) == np.nanmean(reloaded_file)
    target_file.unlink()
    

def test_load_tiff():
    target_file = Path('test.tiff')
    test = np.random.rand(1024,1280,3).astype(np.float32)
    test[200,300,0]=np.nan
    test[100,30]=np.nan
    sio.save_img3d(target_file, test)
    loaded = sio.load_img3d(target_file)
    np.testing.assert_equal(test, loaded)
    target_file.unlink()
    

def test_load_obj():
    ptcloud_obj = sio.load_scared_obj('./data/keyframe_1/point_cloud.obj')
    img3d = sio.load_img3d('./data/keyframe_1/left_depth_map.tiff')
    img3d = img3d.reshape(-1,3)
    img3d = img3d[~np.isnan(img3d).any(axis=1)]
    assert img3d.shape == ptcloud_obj.shape
    assert np.all(img3d == ptcloud_obj)
    assert img3d.dtype == np.float32
    assert ptcloud_obj.dtype == np.float32
    




def test_save_load_ply():
    # test when without nan values
    target_file = Path('test.ply')
    img3d = sio.load_img3d('./data/keyframe_1/left_depth_map.tiff')
    img3d = img3d.reshape(-1,3)
    pt_cloud = img3d[~np.isnan(img3d).any(axis=1)]
    sio.save_ptcloud_as_ply(target_file, pt_cloud, save_binary=True)
    loaded_pt_cloud = sio.load_ply_as_ptcloud(target_file)
    assert loaded_pt_cloud.shape == pt_cloud.shape
    np.testing.assert_equal(pt_cloud, loaded_pt_cloud)
    sio.save_ptcloud_as_ply(target_file, pt_cloud, save_binary=False)
    loaded_pt_cloud = sio.load_ply_as_ptcloud(target_file)
    assert loaded_pt_cloud.shape == pt_cloud.shape
    np.testing.assert_equal(pt_cloud, loaded_pt_cloud)
    target_file.unlink()
    

def test_save_load_subpix_png():
    target_file = Path('test.png')
    
    # test when without nan values
    img3d = sio.load_img3d('./data/keyframe_1/left_depth_map.tiff')
    depthmap = img3d[:,:,2].copy()
    sio.save_subpix_png(target_file, depthmap)
    loaded_depthmap = sio.load_subpix_png(target_file)
    
    # test loaded shape equal with saved shape
    assert loaded_depthmap.shape == depthmap.shape
    
    # test if saving function changed the data in any way
    np.testing.assert_equal(depthmap, img3d[:,:,2])
    assert depthmap.dtype == img3d.dtype
    
    #test if 0 values are loaded as np.nan
    np.testing.assert_equal(np.isnan(depthmap), np.isnan(loaded_depthmap))
    
    # there is drop in accuracy because we store floats as 16bit integers.
    # the two arrays should be the same up for the 2 decimal points.
    np.testing.assert_almost_equal(depthmap, loaded_depthmap, decimal=2)
    
    sio.save_subpix_png(target_file, loaded_depthmap)
    reloaded_depthmap = sio.load_subpix_png(target_file)
    np.testing.assert_array_equal(reloaded_depthmap, loaded_depthmap)
    target_file.unlink()

    