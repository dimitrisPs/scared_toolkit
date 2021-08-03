import numpy as np


def depthmap_coverage(depthmap):
    """computes the proportion of pixels with known ground truth

    Args:
        depthmap (np.ndarray): depthmap with size hxw

    Returns:
        float: the proportion of pixels with known ground truth [0-1]
    """
    depthmap = depthmap.reshape(-1)
    num_of_pixels = depthmap.shape[0]

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
        [np.float]: [proportion of pixels in the ref that ground truth is known]
        [np.float]: [proportion of pixels that information is present both in ref and comp]
    """
    assert ref.shape == comp.shape
    ref[ref == 0] = np.nan
    comp[comp == 0] = np.nan

    # find the proporsion of pixels with ground truth
    coverage = depthmap_coverage(ref)
    if coverage < 0.1:
        return np.nan, coverage

    ref = ref.reshape(-1)
    comp = comp.reshape(-1)

    abs_diff = np.abs(ref-comp)
    error = np.nanmean(abs_diff)

    assessed = np.count_nonzero(~np.isnan(abs_diff)) / (ref.shape[0])

    return error, coverage, assessed


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
        [np.float]: [proportion of pixels in the ref that ground truth is known]
        [np.float]: [proportion of pixels that information is present both in ref and comp]
    """
    assert ref.shape == comp.shape

    # find the proporsion of pixels with ground truth, we one only one channel
    # with depthmap_coverage to check for nan values.
    coverage = depthmap_coverage(ref[:, :, 2])
    if coverage < 0.1:
        return np.nan, coverage, 0
    ref = ref.reshape(-1, 3)
    comp = comp.reshape(-1, 3)
    comp[comp == 0] = np.nan
    distance = np.sqrt(np.sum((ref-comp)**2, axis=1))
    error = np.nanmean(distance)

    assessed = np.count_nonzero(~np.isnan(distance)) / (ref.shape[0])

    return error, coverage, assessed
