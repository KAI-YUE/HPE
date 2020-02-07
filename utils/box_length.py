# -*- coding: utf-8 -*-
"""
The length of the bounding box is determined by L = w_1/depth + w_0;
This script finds w_0/1 with mean square error regression.
"""

# Python Libraries
import os
import cv2
import numpy as np

def get_boundry(img):
    """
    Given a labelled image with hand in the bounding box, 
    returns the coordinates of the boundary.
    """
    box_color = np.array([255, 0, 0])
    skip = 50
    
    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            if (img[i, j] == box_color).all():
                begin_x = i
                begin_y = j
                for k in range(i+skip, img.shape[0], 2):
                    for l in range(j+skip, img.shape[1], 2):
                        if (img[k, l] == box_color).all():
                            end_x = k
                            end_y = l
                            
                            return [begin_x, end_x, begin_y, end_y]
                            
if __name__ == "__main__":
    ori_data_dir = r"D:\YUE\SynthHands\01"
    labeled_data_dir = r"D:\YUE\SynthHands\labelled"
    
    file_list =  os.listdir(labeled_data_dir)
    num_files = len(file_list)
    
    # Find the relationship between length and depth
    box_length = np.zeros(num_files)
    depth_arr = np.zeros(num_files)
    
    for i, f in enumerate(file_list):
        img = cv2.imread(os.path.join(labeled_data_dir, f))[..., ::-1]
        depth = np.loadtxt(os.path.join(ori_data_dir, f[:8]+"_joint_pos.txt"))
        
        depth_arr[i] = depth[0, -1]
        boundry = get_boundry(img)
        box_length[i] = max(boundry[1]-boundry[0], boundry[3]-boundry[2])
     
    x = 1 / depth_arr
    w1 = (num_files * np.sum(x * box_length) - np.sum(x) * np.sum(box_length) ) /   \
         (num_files * np.sum(x**2) - (np.sum(x))**2 )
    
    w0 = (np.sum(box_length) - w1 * np.sum(x)) / num_files
        
        

                            
    
    

