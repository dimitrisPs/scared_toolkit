import numpy as np


def depthmap_coverage(depthmap):
    """computes the proportion of pixels with known ground truth

    Args:
        depthmap (np.ndarray): depthmap with size hxw

    Returns:
        float: the proportion of pixels with known ground truth [0-1]
    """
    
    depthmap = depthmap.reshape(-1)
    num_of_pixels = depthmap.shape
    
    num_of_known_depths = num_of_pixels - np.count_nonzero(np.isnan(depthmap))

    proportion = num_of_known_depths / num_of_pixels
    
    return proportion

def depthmap_error(ref, comp):
    """computes SCARED Frame error from depthmaps.
    
    Compute error between ref and comp depthmap using mean absolute difference
    in 3d, basically mean distance. The function expects floating point hxwx1 arrays,
    with unknown pixels values in ref, set to np.nan

    Args:
        ref ([np.ndarray]): [reference depthmap]
        comp ([np.ndarray]): [depthmap to compare with reference]

    Returns:
        [np.float]: [error ref and comp, np.nan if coverage is below 10% of total pixels]
    """
    assert ref.shape == comp.shape
    
    # find the proporsion of pixels with ground truth
    if depthmap_coverage(ref) < 0.1:
        return np.nan
    
    ref = ref.reshape(-1)
    comp = comp.reshape(-1)
    abs_diff = np.abs(ref-comp)
    error = np.nanmean(abs_diff)
    
    return error


def xyz_error(ref, comp):
    """computes SCARED Frame error.
    
    Compute error between ref and comp xyz images, using mean absolute difference.
    The function expects floating point hxwx3 imputs arrays, with unknown pixels
    values in ref, set to np.nan. Essentially this compares directly with the 
    ground truth provided by the authors and because of inacuracy in calibration
    parameters the error will be different from assesing on depthmaps.

    Args:
        ref ([np.ndarray]): [reference xyz]
        comp ([np.ndarray]): [xyz to compare with reference]

    Returns:
        [np.float]: [error ref and comp, np.nan if coverage is below 10% of total pixels]
    """
    assert ref.shape == comp.shape
    
    # find the proporsion of pixels with ground truth, we one only one channel
    # with depthmap_coverage to check for nan values.
    if depthmap_coverage(ref[:,:,2]) < 0.1:
        return np.nan
    
    ref = ref.reshape(-1,3)
    comp = comp.reshape(-1,3)
    distance = np.sqrt(np.sum((ref-comp)**2,axis=1))
    error = np.nanmean(distance)
    return error