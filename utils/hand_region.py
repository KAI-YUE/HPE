# Python Libraries
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
The paramters of the trained SVM. Run 'skin_classifier.py' first.
"""
skin_w = np.array([0.70, 8.45, 22.97])
skin_b = -16.73
fg_w = -0.0041
fg_b = 1.78

def to_YCbCr(img):
    """
    convert a float image ([0, 1]) to YCbCr format.
    """
    delta = 128/255
    YCbCr = np.zeros_like(img)
    YCbCr[...,0] = 0.299*img[..., 0] + 0.587*img[...,1] + 0.114*img[...,2]
    YCbCr[...,1] = 0.546*(img[..., 2] - YCbCr[...,0]) + delta
    YCbCr[...,2] = 0.713*(img[...,0] - YCbCr[...,0]) + delta
    
    return YCbCr

def skin_mask(YCbCr):
    """
    Generate a skin mask.
    """
    sign_map = skin_w[0] * YCbCr[...,0] + skin_w[1] * YCbCr[...,1] + skin_w[2] * YCbCr[..., 2] + skin_b
    mask = (sign_map>0)
    
    # Reduce the influence of background noise
    return mask

def foreground_mask(depth):
    """
    Generate the foreground mask.
    """   
    # Based on the result of SVM regression.
    return (depth<434) * (depth>0)

def num_skin_pixels(mask):
    """
    Reurns the number of skin pixels in the img.
    ---------------------------------------------------
    Args,
        mask,   the skin mask
    """                     
    return np.sum(mask) 


def ROI_Hand(img, depth, center, invalid_depth=32001):
    """
    Extract the ROI containing the hand, i.e, the bounding box.
    ---------------------------------------------------------------------------------
    Args,
        img:            ndarray, the original RGB image. [0, 1]
        depth:          ndarray, the depth image
        center:         tuple/ndarray/list, [x->col, y->row] the center of the root.
        invalid_depth:  the value of the invald depth
    Retruns,
        bound:      [x_begin, x_end, y_begin, y_end]
    """
    size_factor = 27025 
    mean_depth = 0
    num_valid = 0

    search_region = 2
    i_min = max(0, center[1]-search_region)
    i_max = min(depth.shape[0], center[1]+search_region+1)
    j_min = max(0, center[0]-search_region)
    j_max = min(depth.shape[1], center[0]+search_region+1)
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if depth[i,j] != invalid_depth:
                mean_depth += depth[i,j]
                num_valid += 1
    
    if num_valid > 0:
        mean_depth /= num_valid
        radius = size_factor / mean_depth 

        x_begin = max(center[1] - radius, 0)
        x_end = min(center[1] + radius, img.shape[0]-1)
        y_begin = max(center[0] - radius, 0)
        y_end = min(center[0] + radius, img.shape[1]-1)
    
        return dict(ROI = np.array([x_begin, x_end, y_begin, y_end], dtype="int16"), 
                    mean_depth = mean_depth)
     

def ROI_from_pos(pos_arr, size=128):
    """
    Extract the ROI containing the hand, given the 2d pos array.
    """
    left_bound = np.min(pos_arr[:,0])
    right_bound = np.max(pos_arr[:,0])
    up_bound = np.min(pos_arr[:,1])
    bottom_bound = np.max(pos_arr[:,1])
    
    if right_bound - left_bound <=size:
        margin = size - right_bound + left_bound
        if margin%2 == 0:
            half_margin = 0.5*margin
            left_bound -= half_margin
            right_bound += half_margin
        else:
            half_margin = 0.5*(margin-1)
            left_bound -= half_margin + 1
            right_bound += half_margin
    
    if bottom_bound - up_bound <=size:
        margin = size - bottom_bound + up_bound
        if margin%2 == 0:
            half_margin = int(0.5*margin)
            up_bound -= half_margin
            bottom_bound += half_margin
        else:
            half_margin = int(0.5*(margin-1))
            up_bound -= half_margin + 1
            bottom_bound += half_margin
    
    return np.asarray([up_bound, bottom_bound, left_bound, right_bound], dtype="int")

#if __name__ == "__main__":
#    data_path = r"F:\DataSets\Transformed_SynthHands\female_noobject\seq02\cam01\01\00000062.dat"
#    pos_path = data_path.replace(".dat", "_joint_pos.txt")
#    
#    with open(data_path, "rb") as fp:
#        a_set = pickle.load(fp)
#    img = a_set['img']
#    depth = a_set['depth']
#    
#    pos_arr = np.loadtxt(pos_path)
#    
#    ROI = ROI_Hand(img, depth, pos_arr[21, :-1].astype(int))
#    
#    fig = plt.figure()
#    ax = plt.gca()
#    ax.add_patch(plt.Rectangle([ROI[2], ROI[0]], ROI[3]-ROI[2], ROI[1]-ROI[0], fill=False, color='r'))
#    ax.imshow((255*img).astype("uint8"))


