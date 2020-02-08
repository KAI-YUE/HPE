"""
Transform the original SynthHands Dataset.
    1. Original images (color on depth) are downsampled. 
    2. The 3d coordinates in the depth camera frame are transferred to 2d.
    3. Gaussian heatmap of different parts will be stored.
"""

# Python Libraries
import os
import cv2
import pickle
import numpy as np

# My Libraries
from utils.heatmap import Gaussian_heatmap
from utils.hand_region import ROI_Hand    

f_x = 475.62
f_y = 475.62
x_0 = 311.125
y_0 = 245.965

num_parts = 21

Part_Namelist = ["W", 
"T0", "T1", "T2", "T3", 
"I0", "I1", "I2", "I3",
"M0", "M1", "M2", "M3",
"R0", "R1", "R2", "R3",
"L0", "L1", "L2", "L3"]


def trans_for_HLoNet(root_dir, dst_dir):
    for root, dirs, files in os.walk(root_dir):
        
        new_path = root.replace(root_dir, dst_dir)
        if (not os.path.exists(new_path)):
            os.mkdir(new_path)    
        
        if files != []:
            for f in files:
                    
                if 'joint_pos' in f:
                    with open(os.path.join(root, f)) as fp:
                        line = fp.readline()
                    
                    pos_coord = line.split(',')
                    pos = np.zeros((num_parts, 3))
                    transferred_pos = np.zeros((num_parts, 3))
                    
                    for i in range(num_parts):
                        for j in range(3):
                            pos[i][j] = float(pos_coord[3 * i + j])

                    # Downsample by multiplying 0.5
                    for i in range(num_parts):
                        transferred_pos[i, 0] = int(0.5*(f_x / pos[i, 2] * pos[i, 0] + x_0))
                        transferred_pos[i, 1] = int(0.5*(f_y / pos[i, 2] * pos[i, 1] + y_0))
                    
                    np.savetxt(os.path.join(new_path, f), np.vstack((pos, transferred_pos)))
                    
                    img_name = f.replace("joint_pos.txt", "color_on_depth.png")
                    color_on_depth = cv2.imread(os.path.join(root, img_name))
                    [h, w] = [int(0.5*color_on_depth.shape[0]), int(0.5*color_on_depth.shape[1])]
                    color_on_depth = cv2.resize(color_on_depth, (w, h))
                    
                    
                    # Save img, pos and all of the heatmap ground truth in a dict 
                    dat_dict = {"img":(color_on_depth[:,:,::-1]/255).astype(np.float16)}
                    
                    # Read depth image 
                    depth_name = f.replace("joint_pos.txt", "depth.png")
                    depth = cv2.imread(os.path.join(root, depth_name), -1)
                    dat_dict['depth'] = cv2.resize(depth, (w, h))
                    
                    wrist_set = Gaussian_heatmap(color_on_depth, transferred_pos[0, :2])
                    cv2.imwrite(os.path.join(new_path, f.replace("joint_pos.txt", "composite.png")), wrist_set["composite"])
                    dat_dict["W_ori"] = cv2.resize(wrist_set["confidence_map"], (80, 60)).astype(np.float16)
                        
                    with open(os.path.join(new_path, f.replace("_joint_pos.txt", ".dat")), 'wb') as fp:
                        pickle.dump(dat_dict, fp)

def trans_for_PReNet(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if files != []:
            for f in files:
                if ".dat" in f:
                    with open(os.path.join(root, f), "rb") as fp:
                        a_set = pickle.load(fp)
                    
                    f_ = f.replace(".dat", "_joint_pos.txt")
                    pos_arr = np.loadtxt(os.path.join(root, f_))
                    box = ROI_Hand(a_set['img'], a_set['depth'], pos_arr[21, :2].astype("int"))
                    cropped_img = (255*a_set['img'][box[0]:box[1], box[2]:box[3]]).astype("uint8")
                    
                    hm = np.zeros((21,64,64), dtype="float16")
                    for i in range(num_parts):             
                        results = Gaussian_heatmap(cropped_img, 
                                                   [int(pos_arr[21+i, 0]-box[2]), 
                                                    int(pos_arr[21+i, 1]-box[0])])
    
                        hm[i] = cv2.resize(results["confidence_map"], (64, 64)).astype(np.float16)
                    
                    a_set["heatmaps"] = hm
                    a_set["hand"] = (cv2.resize(cropped_img, (128, 128))/255).astype(np.float16)
                    a_set["pos_arr"] = (pos_arr[:21] - pos_arr[0])
    
                    with open(os.path.join(root, f), "wb") as fp:
                        pickle.dump(a_set, fp)




if __name__ == "__main__":
    root_dir = r"F:\DataSets\SynthHands_Release\male_object\seq07\cam05"
#    dst_dir = r"F:\\Datasets\\Transformed_SynthHands"
    dst_dir = r"F:\DataSets\SynthHands_toy"

    trans_for_HALNet(root_dir, dst_dir)
    trans_for_JORNet(dst_dir)

                