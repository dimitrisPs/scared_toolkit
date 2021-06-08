from scareddataset import convertions as cvt
from scareddataset import io as sio
import numpy as np
import json

def read_frame_data(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

servct_calib = read_frame_data('./data/servdata/001_calib.json')


def test_accuracy_ptcloud_to_img3d():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    
    ref_pt_cloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_ptcloud_shuffled = ref_pt_cloud.copy()

    gen_img3d = cvt.ptcloud_to_img3d(ref_pt_cloud,K,np.zeros(5),(576,720))
    gen_img3d_suf = cvt.ptcloud_to_img3d(ref_ptcloud_shuffled,K,np.zeros(5),(576,720))
    
    gen_pt_cloud = gen_img3d.reshape(-1,3).copy()
    gen_pt_cloud_suf = gen_img3d_suf.reshape(-1,3).copy()
    
    assert gen_pt_cloud.shape == ref_pt_cloud.shape
    np.testing.assert_array_almost_equal(ref_pt_cloud, gen_pt_cloud, decimal=5)
    np.testing.assert_array_almost_equal(ref_pt_cloud, gen_pt_cloud_suf, decimal=5)
    
def test_accuracy_ptcloud_to_disparity():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    P1 = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)
    P2 = np.array(servct_calib['P2']['data'], dtype=np.float64).reshape(3,4)
    
    ref_pt_cloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    gen_disparity = cvt.ptcloud_to_disparity(ref_pt_cloud, P1, P2, ref_disparity.shape)
    
    
    assert ref_disparity.shape == gen_disparity.shape
    np.testing.assert_array_almost_equal(ref_disparity, gen_disparity, decimal=5)
    
def test_accuracy_ptcloud_to_depthmap():
    
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    
    ref_pt_cloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    gen_depthmap = cvt.ptcloud_to_depthmap(ref_pt_cloud, K, np.zeros(5), ref_depthmap.shape)
    
    
    assert ref_depthmap.shape == gen_depthmap.shape
    np.testing.assert_array_almost_equal(ref_depthmap, gen_depthmap, decimal=2)
    
    
    
    
def test_accuracy_disparity_to_ptcloud():
    assert 'Q' in servct_calib
    assert servct_calib['Q'] is not None
    Q = np.array(servct_calib['Q']['data'], dtype=np.float64).reshape(4,4)
    
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    gen_ptcloud = cvt.disparity_to_ptcloud(ref_disparity, Q)
    assert gen_ptcloud.shape == ref_ptcloud.shape
    np.testing.assert_array_almost_equal(ref_ptcloud, gen_ptcloud, decimal=5)
    
    


def test_accuracy_disparity_to_img3d():
    assert 'Q' in servct_calib
    assert servct_calib['Q'] is not None
    Q = np.array(servct_calib['Q']['data'], dtype=np.float64).reshape(4,4)
    
    
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    h,w= ref_disparity.shape
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_img3d = ref_ptcloud.reshape(h,w,-1) # points are stored in continues manner.
    gen_img3d = cvt.disparity_to_img3d(ref_disparity, Q)

    assert ref_img3d.shape == gen_img3d.shape
    np.testing.assert_array_almost_equal(gen_img3d, ref_img3d, decimal=5)

def test_accuracy_disparity_to_depthmap():
    assert 'Q' in servct_calib
    assert servct_calib['Q'] is not None
    Q = np.array(servct_calib['Q']['data'], dtype=np.float64).reshape(4,4)
    
    
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    
    gen_depthmap = cvt.disparity_to_depthmap(ref_disparity, Q)
    
    assert ref_depthmap.shape == gen_depthmap.shape
    
    np.testing.assert_array_almost_equal(ref_depthmap, gen_depthmap, decimal=2)

    
def test_accuracy_img3d_to_ptcloud():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_img3d = cvt.ptcloud_to_img3d(ref_ptcloud, K, np.zeros(5), (576,720))
    gen_pt_cloud = cvt.img3d_to_ptcloud(ref_img3d)
    np.testing.assert_array_almost_equal(ref_ptcloud, gen_pt_cloud, decimal=5)
    
    
def test_accuracy_img3d_to_depthmap():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_img3d = cvt.ptcloud_to_img3d(ref_ptcloud, K, np.zeros(5), (576,720))
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    gen_depthmap = cvt.img3d_to_depthmap(ref_img3d)
    np.testing.assert_array_almost_equal(ref_depthmap, gen_depthmap, decimal=2)

def test_accuracy_img3d_to_disparity():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    P1 = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)
    P2 = np.array(servct_calib['P2']['data'], dtype=np.float64).reshape(3,4)
    K = P1[:,:3]
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_img3d = cvt.ptcloud_to_img3d(ref_ptcloud, K, np.zeros(5), (576,720))
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    gen_disparity = cvt.img3d_to_disparity(ref_img3d, P1, P2)
    np.testing.assert_array_almost_equal(ref_disparity, gen_disparity, decimal=5)
    
    
def test_accuracy_depthmap_to_disparity():
    assert 'Q' in servct_calib
    assert servct_calib['Q'] is not None
    Q = np.array(servct_calib['Q']['data'], dtype=np.float64).reshape(4,4)
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    ref_disparity = sio.load_subpix_png('./data/servdata/001_disparity.png')
    gen_disparity = cvt.depthmap_to_disparity(ref_depthmap, Q)
    
    np.testing.assert_array_almost_equal(ref_disparity, gen_disparity, decimal=2)
    
def test_accuracy_depthmap_to_img3d():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_img3d = cvt.ptcloud_to_img3d(ref_ptcloud, K, np.zeros(5), (576,720))
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    gen_img3d = cvt.depthmap_to_img3d(ref_depthmap, K)
    
    
    np.testing.assert_array_almost_equal(gen_img3d, ref_img3d, decimal=2)    
    
def test_accuracy_depthmap_to_ptcloud():
    assert 'P1' in servct_calib
    assert servct_calib['P1'] is not None
    K = np.array(servct_calib['P1']['data'], dtype=np.float64).reshape(3,4)[:,:3]
    ref_ptcloud = sio.load_ply_as_ptcloud('./data/servdata/001_ptcloud.ply')
    ref_depthmap = sio.load_subpix_png('./data/servdata/001_depthmap.png')
    gen_ptcloud = cvt.depthmap_to_ptcloud(ref_depthmap, K)
    
    
    np.testing.assert_array_almost_equal(gen_ptcloud, ref_ptcloud, decimal=2)    
            
