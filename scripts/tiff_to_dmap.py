import tifffile as tiff
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='convert ground truth from Scared format to depthmaps.')
parser.add_argument('input', help='path to scared depthmap tiff file.')
parser.add_argument('output', help='path to store the resulting depthmap.')


def load_depthmap(path):
    depthmap = tiff.imread(str(path))
    depthmap[depthmap==0]=np.nan
    return depthmap.astype(np.float32)[:,:,-1]


if __name__ == "__main__":
    args = parser.parse_args()
    
    scared_depthmap = load_depthmap(args.input)   
    
    cv2.imwrite(str(args.output), (scared_depthmap*256).astype(np.uint16))