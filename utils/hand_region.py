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
    convert a float image ([]0, 1]) to YCbCr format.
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
    return (depth < 434)

def num_skin_pixels(mask):
    """
    Reurns the number of skin pixels in the img.
    ---------------------------------------------------
    Args,
        mask,   the skin mask
    """                     
    return np.sum(mask) 

def ROI_Hand(img, depth, center):
    """
    Find the ROI containing the hand, i.e, the bounding box.
    ---------------------------------------------------------------------------------
    Args,
        img:        ndarray, the original RGB image. [0, 1]
        depth:      mdarray, the depth image
        center:     tuple/ndarray/list, [x->col, y->row] the center of the wrist.
    Retruns,
        bound:      [x_begin, x_end, y_begin, y_end] (x_end-x_begin=128, y_end-y_begin=128)
    """
    
    limit = 112
    step = 1
    th = 3
    margin = 8
    
    y_begin = int(max(center[0] - limit,0))
    y_end = int(min(center[0] + limit, img.shape[1]-1))
    
    YCrCb = to_YCbCr(img)
    skin_m = skin_mask(YCrCb)
    fg_m = foreground_mask(depth)
    
    mask = skin_m * fg_m
#    mask = foreground_mask(depth)
    
    x_begin = max(center[1] - limit, 0)
    for i in range(x_begin, center[1], step):
        if num_skin_pixels(mask[i, y_begin:y_end]) >= th:
            x_begin = i
            break
    
    x_end = min(x_begin+limit, img.shape[0]-1)
    
    left_pixels = 0
    right_pixels = 0
    for j in range(y_end, center[0], -step):
        if num_skin_pixels(mask[x_begin:x_end, j]) >= th:
            j_end = min(j+margin, img.shape[1]-1)
            right_pixels = num_skin_pixels(mask[x_begin:x_end, j-margin:j_end])
            y_end = j
            break
    
    for j in range(y_begin, center[0], step):
        if num_skin_pixels(mask[x_begin:x_end, j]) >= th:
            j_begin = max(j-margin, img.shape[0])
            left_pixels = num_skin_pixels(mask[x_begin:x_end, j_begin:j+step])
            y_begin = j
            break
    
    if left_pixels > right_pixels:
        y_begin = y_end - limit
    else:
        y_end = y_begin + limit
           
    return [max(x_begin-2*margin, 0), min(x_end, img.shape[0]-1),  
            max(y_begin-margin, 0), min(y_end+margin, img.shape[1]-1)]
            
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


