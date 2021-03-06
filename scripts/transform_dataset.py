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
# from utils.heatmap import Gaussian_heatmap
# from utils.hand_region import ROI_Hand, ROI_from_pos    

f_x = 475.62
f_y = 475.62
x_0 = 311.125
y_0 = 245.965

max_depth = 1000
root_index = 9
num_parts = 21

Part_Namelist = ["W", 
"T0", "T1", "T2", "T3", 
"I0", "I1", "I2", "I3",
"M0", "M1", "M2", "M3",
"R0", "R1", "R2", "R3",
"L0", "L1", "L2", "L3"]

def trans_SynthHands(src_dir, dst_dir):
    """
    Transform the synthHands dataset to .dat files.
    --------------------------------------------------------------------
    Args,
        src_dir:    the root directory of the synthhand dataset.
        dst_dir:    the directory of the transfromed dataset.
    """
    f_x = 475.62
    f_y = 475.62
    x_0 = 311.125
    y_0 = 245.965

    num_parts = 21
    root_index = 9
    crop_size = 128
    depth_max = 1000
    pos_scale = 1000
    template = np.zeros((crop_size, crop_size, 3), dtype="uint8")

    for root, dirs, files in os.walk(src_dir):
        
        print(root)
        
        new_path = root.replace(src_dir, dst_dir)
        if (not os.path.exists(new_path)):
            os.mkdir(new_path)    
            
        
        if files != []:
            for f in files:
                if 'joint_pos' in f:
                    with open(os.path.join(root, f)) as fp:
                        line = fp.readline()
                    
                    pos_coord = line.split(',')
                    pos = np.zeros((num_parts, 3), dtype="float16")
                    projected_pos = np.zeros((num_parts, 2), dtype="uint8")
                    
                    for i in range(num_parts):
                        for j in range(3):
                            pos[i][j] = float(pos_coord[3 * i + j])
                            
                    # Downsample by multiplying 0.5
                    for i in range(num_parts):
                        projected_pos[i, 0] = 0.5*(f_x / pos[i, 2] * pos[i, 0] + x_0)
                        projected_pos[i, 1] = 0.5*(f_y / pos[i, 2] * pos[i, 1] + y_0)
                        
                    img_name = f.replace("joint_pos.txt", "color_on_depth.png")
                    color_on_depth = cv2.imread(os.path.join(root, img_name))
                    [h, w] = [int(0.5*color_on_depth.shape[0]), int(0.5*color_on_depth.shape[1])]
                    color_on_depth = cv2.resize(color_on_depth, (w, h))
                    
                    # Save img, pos and all of the heatmap ground truth in a dict 
                    dat_dict = {"img":(color_on_depth[:,:,::-1]/255).astype(np.float16)}
                    
                    # Read depth image 
                    depth_name = f.replace("joint_pos.txt", "depth.png")
                    depth = cv2.imread(os.path.join(root, depth_name), -1)
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                    depth_min = np.min(depth)
                    
                    dat_dict['depth'] = depth
                    depth_norm = np.where(depth>depth_max, 1, (depth - depth_min)/(depth_max-depth_min))
                    dat_dict['depth_norm'] = depth_norm.astype("float16")
                    
                    a_set = Gaussian_heatmap(color_on_depth, projected_pos[root_index])
                    dat_dict["root_hm"] = a_set["confidence_map"].astype(np.float16)
                    
                    # ROI = ROI_from_pos(projected_pos)
                    ROI, mean_depth = ROI_Hand(color_on_depth, depth, projected_pos[root_index])
                    
                    if (ROI[1] - ROI[0] > crop_size):
                        row_scale = crop_size / (ROI[1] - ROI[0]) 
                    else:
                        row_scale = 1
                        
                    if (ROI[3] - ROI[2] > crop_size):
                        col_scale = crop_size / (ROI[3] -ROI[2]) 
                    else:
                        col_scale = 1
                    
                    dat_dict["scale_factors"] = np.array([row_scale, col_scale], dtype="float16")
                    
                    hms = np.zeros((21, crop_size, crop_size), dtype="float16")
                    for i in range(num_parts):             
                        results = Gaussian_heatmap(template, 
                                                [int((projected_pos[i, 0]-ROI[2])*col_scale), 
                                                    int((projected_pos[i, 1]-ROI[0])*row_scale)])
                        hms[i] = results["confidence_map"]
                    
                    cropped_img= cv2.resize(color_on_depth[ROI[0]:ROI[1], ROI[2]:ROI[3]], 
                                                        (crop_size,crop_size))
                    dat_dict["cropped_img"] = (cropped_img[...,::-1]/255).astype("float16")
                    
                    cropped_depth = cv2.resize(depth[ROI[0]:ROI[1], ROI[2]:ROI[3]], (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                    cropped_depth = np.where(cropped_depth>depth_max, 1, (cropped_depth - mean_depth) / depth_max)
                    dat_dict["cropped_depth"] = cropped_depth.astype("float16")
                    
                    dat_dict["heatmaps"] = hms
                    dat_dict["root_pos"] = pos[root_index].copy()
                    
                    # Calculate the rotation matrix for normalization
                    pos -= pos[0]
                    _05_vec = pos[5] 
                    _09_vec = pos[9]
                    z_body_frame = np.cross(_05_vec, _09_vec)
                    
                    # Normalize the y axis and z axis in the body frame
                    y_body_frame = _09_vec / np.linalg.norm(_09_vec)
                    z_body_frame = z_body_frame / np.linalg.norm(z_body_frame)
                    x_body_frame = np.cross(y_body_frame, z_body_frame).reshape(-1,1)
                    
                    y_body_frame = y_body_frame.reshape((-1,1))
                    z_body_frame = z_body_frame.reshape((-1,1))
                    
                    R = np.array([[0,0,1],[1,0,0],[0,1,0]]) @ \
                        np.hstack((y_body_frame, z_body_frame, x_body_frame)).T
                        
                    norm_3d_pos = np.zeros_like(pos)
                    for i in range(1, norm_3d_pos.shape[0]):
                        norm_3d_pos[i] = R @ pos[i]
                        
                    a_set["norm_3d_pos"] = norm_3d_pos.astype("float16")
                        
                    pos -= pos[root_index]
                    dat_dict["3d_pos"] = (pos/pos_scale).astype("float16")
                    dat_dict["2d_pos"] = projected_pos
                    dat_dict["ROI"] = ROI.astype("int16")
                    
                    with open(os.path.join(new_path, f.replace("_joint_pos.txt", ".dat")), 'wb') as fp:
                        pickle.dump(dat_dict, fp)


def trans_EgoData(src_dir, dst_dir, category="Desk"):
    """
    Transform the EgoDexter dataset into .dat file.
    ---------------------------------------------------------------------------
    Args,
        src_dir:    the sub-directory of the egoDexter dataset.
        dst_dir:    the destination directory of the transformed dataset.
        category:   the specific category. (Desk, Fruits, Kitchen, Rotunda)
    """
    invalid_depth = 0
    depth_max = 1000
    scale_factor = 2
    
    color_file = "image_{:05d}_color_on_depth.png"
    depth_file = "image_{:05d}_depth.png"
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    pos2d_file = os.path.join(src_dir, "annotation.txt") 
    pos3d_file = os.path.join(src_dir, "annotation.txt_3D.txt")

    with open(pos2d_file) as fp:
        pos2d_strings = fp.readlines()
        
    with open(pos3d_file) as fp:
        pos3d_strings = fp.readlines()

    for i in range(len(pos2d_strings)):
        pos_2d = str2arr(pos2d_strings[i], "2d")
        pos_3d = str2arr(pos3d_strings[i], "3d")
        
        if (pos_2d == -1).any() or (pos_3d == 0).any():
            continue
        
        else:
            a_set = {}
            color_on_depth = cv2.imread(os.path.join(src_dir, "color_on_depth", color_file.format(i)))[...,::-1]
            color_on_depth = cv2.resize(color_on_depth, (320, 240))
            depth = cv2.imread(os.path.join(src_dir, "depth", depth_file.format(i)), -1)
            depth = cv2.resize(depth, (320, 240))
            
            depth_norm = np.where(depth==invalid_depth, 2*depth_max, depth)
            depth_min = np.min(depth_norm)
            depth_norm = np.where(depth_norm>depth_max, depth_max, depth)
            depth_norm = (depth_norm - depth_min) / (depth_max - depth_min)
            
            a_set["img"] = (color_on_depth/255).astype("float16")
            a_set["depth"] = depth
            a_set["depth_norm"] = depth_norm
            a_set["2d_pos"] = pos_2d/scale_factor
            a_set["3d_pos"] = pos_3d
            
            with open(os.path.join(dst_dir, "{:05d}.dat".format(i)), "wb") as fp:
                pickle.dump(a_set, fp)


def str2arr(string, mode="2d"):
    if mode == "2d":
        arr_list = string.split(";")
        
        if len(arr_list) < 6:
            return np.array(-1)
        
        pos2d_arr = np.zeros((5, 2), dtype="int")
        for i in range(5):
            coord = arr_list[i].split(",")
            pos2d_arr[i,0] = int(coord[0])
            pos2d_arr[i,1] = int(coord[1])
        
        return pos2d_arr
    
    elif mode == "3d":
        arr_list = string.split(";")
        if len(arr_list) < 6:
            return np.array(-1)
        
        pos3d_arr = np.zeros((5, 3))
        
        for i in range(5):
            coord = arr_list[i].split(",")
            pos3d_arr[i,0] = float(coord[0])
            pos3d_arr[i,1] = float(coord[1])
            pos3d_arr[i,2] = float(coord[2])
        
        return pos3d_arr

def get_3dpos(file_name):
    
    num_parts = 21
    
    with open(file_name) as fp:
        line = fp.readline()
    
    pos_coord = line.split(',')
    pos = np.zeros((num_parts, 3), dtype="float16")
        
    for i in range(num_parts):
        for j in range(3):
            pos[i][j] = float(pos_coord[3 * i + j])
    
    return pos

if __name__ == "__main__":
    categories = ["Desk", "Fruits", "Kitchen", "Rotunda"]
    for category in categories:
        src_dir = r"F:\DataSets\EgoDexter\data\{}".format(category)
        dst_dir = r"F:\Datasets\Dexter_transformed\{}".format(category)
        
        trans_EgoData(src_dir, dst_dir, category)