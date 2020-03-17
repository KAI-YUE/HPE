"""
Helpful functions set.
"""

# Python Libraries
import os
import cv2
import pickle
import numpy as np

# Pytorch Libraries
import torch
import torch.nn as nn

# My Libraries
from src.loadConfig import loadConfig
from utils.heatmap import Heatmap 

def post_process(image):
    """
    Convert a tensor image to ndarray in RGB format (ranges between [0, 255]).
    -----------------------------------------------------
    Args,
        image:    tensor (c x H x W), an image in tensor format.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    return (image*std + mean).numpy()


def pos_from_heatmap(heatmaps, depth, ROI):
    """
    Get the pose array from the heatmap.
    -------------------------------------------------------------------------------
    Args,  
        heatmaps:    ndarray([21 x H x W]), the heatmap corresponding to the RGB image. 
        depth:       ndarray([H x W]), the depth image.
        ROI:         thr ROI coordinates of the hand corresponding to the original image. 
    Returns,
        pos_arr:    ndarray(21 x 2 ), each row corresponds to [col, row] (for plot purpose)
    """ 
    pos_arr = np.zeros((21, 2), dtype="int")
    diffident_list = []
    confident_th = 0.5
    num_clusters = 8                        # number of cluster centers to consider

    # Rigister all of the certain keypoints
    for i in range(heatmaps.shape[0]):
        conf_index = np.argmax(heatmaps[i])
        (v, u) = np.unravel_index(conf_index, (heatmaps.shape[1], heatmaps.shape[2]))
        
        if (heatmaps[i, v, u] > confident_th):
            pos_arr[i, 0] = u
            pos_arr[i, 1] = v
        
        else:                                   # have diffident multiple cluster centers
            diffident_list.append(i)

    # Get the cluster centers in the heatmap and get the center with the least error
    for i in diffident_list:
        indices = np.argpartition(heatmaps[i].ravel(), -num_clusters)[-num_clusters:]

        error_arr = np.zeros(num_clusters)
        for j, index in enumerate(indices):
        
            (v, u) = np.unravel_index(index, (heatmaps.shape[1], heatmaps.shape[2]))
            (x1, y1, z1) = back_project((u+ROI[2], v+ROI[0]), depth[v, u])
        
            for k, l in enumerate(links_dict[str(i)]):
                if not l in diffident_list:
                    (x2, y2, z2) = back_project((pos_arr[l,0]+ROI[2], pos_arr[l,1]+ROI[0]), depth[pos_arr[l,1], pos_arr[l,0]])
                    phalanx_length = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                    error_arr[j] += (phalanx_length - mean_dict[str(i)][k])**2
            
        center_index = np.argmin(error_arr)    
        (v, u) = np.unravel_index(indices[center_index], (heatmaps.shape[1], heatmaps.shape[2]))
        pos_arr[i, 0] = u
        pos_arr[i, 1] = v

    return pos_arr


def naive_pos_from_heatmap(heatmaps):
    pos_arr = np.zeros((21, 2), dtype="int")
    
    # Rigister all of the certain keypoints
    for i in range(heatmaps.shape[0]):
        conf_index = np.argmax(heatmaps[i])
        (v, u) = np.unravel_index(conf_index, (heatmaps.shape[1], heatmaps.shape[2]))
        pos_arr[i, 0] = u
        pos_arr[i, 1] = v

    return pos_arr


def center_from_heatmap(wrist_hm):
    """
    Get the hand center from the tensor heatmap (output of the HAL network.)
    --------------------------------------------------------------------------------------
    Args,
        wrist_hm,  tensor(h x w [30 x 40]), the heatmap tensor.
    Returns,
        center:    list, [x->col, y->row].  
    """
    center = [0,0]
    index = torch.argmax(wrist_hm)
    center[1] = int(index // wrist_hm.shape[1] * 240/wrist_hm.shape[0])
    center[0] = int(index % wrist_hm.shape[1] * 320/wrist_hm.shape[1])

    return center


def save_sample(image, gt_heatmap, output, epoch=0, dir="./val_samples"):
    """
    Save the samples for validation.
    --------------------------------------------
    Args,
        image:        tensor (N x c x H x W), a batch of hand images,
        gt_heatmap:   tensor (N x 1 x H x W), ground truth of the confidence map. 
        output:       tensor (N x 1 x H x W), predicted confidence map.
        epoch:        integer. The training epoch used for file name. 
    """

    for i in range(image.shape[0]):
        img = post_process(image[i, ...].cpu().detach())
        img = 255*img.transpose(1,2,0)

        heatmap = gt_heatmap[i, 0, ...].cpu().detach().numpy()
        heatmap = Heatmap(heatmap)

        pred = output[i, 0, ...].cpu().detach().numpy()
        pred = Heatmap(pred)

        alpha = 0.6
        output_image = np.hstack((alpha*img + (1-alpha)*heatmap, alpha*img + (1-alpha)*pred))

        cv2.imwrite(os.path.join(dir, "sample_{}_{}.jpg".format(epoch, i)), output_image)


def save_model(file_name, model, optimizer, epoch=0, max_save=5):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch), file_name)

    # Check the model and only preserve the last max_save(default:5) models
    config = loadConfig()
    flist = os.listdir(config.model_dir)
    if (len(flist) > max_save):
        for f in flist:
            if ('epoch' + str(epoch-max_save*config.save_epoch) in f):
                os.remove(os.path.join(config.model_dir, f))
                break


def project2plane(pos_3d, scale_factor=0.5):
    """
    Project the 3d position array to 2d plane. Camera parameters: f_x = 475.6 f_y = 475.62 x_0 = 311.125 y_0 = 245.965
    -------------------------------------------------------------------------
    Args,
        pos_3d:    ndarray(21 x 3), the 3d position array. 
        scale_factor:   the scale factor of the 2d_positions.
    """
    f_x = 475.62
    f_y = 475.62
    x_0 = 311.125
    y_0 = 245.965

    num_parts = 21
    pos_arr = np.zeros((num_parts, 2), dtype="int")

    for i in range(num_parts):
        pos_arr[i, 0] = 0.5 * (f_x / pos_3d[i, 2] * pos_3d[i, 0] + x_0)
        pos_arr[i, 1] = 0.5 * (f_y / pos_3d[i, 2] * pos_3d[i, 1] + y_0)

    return pos_arr

def back_project(pos_2d, depth, scale_factor=2, invalid_depth=0):
    """
    Back project the 2d coordinates to the 3d frame. x = d/f_x * (u-x0), y = d/f_y * (v-y0). 
    ---------------------------------------------------------------------------
    Args,
        pos_2d:         ndarray(2, ), (u, v)
        depth:          the depth map.
        scale_factor:   the scale factor of the 2d_positions.
    Returns,
        pos_3d:    ndarray(3, ), (x, y, z)
    """
    f_x = 475.62
    f_y = 475.62
    x_0 = 311.125
    y_0 = 245.965

    search_region = 5
    pos_3d = np.array([0, 0, 0])
    pos_3d[0] = depth/f_x * (scale_factor * pos_2d[0] - x_0)
    pos_3d[1] = depth/f_y * (scale_factor * pos_2d[1] - y_0)

    sum_depth = 0
    valid_counter = 0
    i_min = max(0, pos_2d[1])
    i_max = min(depth.shape[0], pos_2d[1])
    j_min = max(0, pos_2d[0])
    j_max = min(depth.shape[1], pos_2d[0])
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if depth[i,j] != invalid_depth:
                sum_depth += depth[i,j]
                valid_counter += 1

    pos_3d[2] = sum_depth/valid_counter
    return pos_3d


def freeze_layers(model, indices=None):
    """
    Freeze the specific parameters of the network layers.
    -----------------------------------------------------------------
    Args,
        indices:    the indices of the layer to be froze. 
                    Default None value will freeze all layers. 
    """
    if indices == None:
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
    else:
        i = -1
        j = 0
        for child in model.children():
            if j == len(indices):
                break
            index = indices[j]
            i += 1
            if i == index:
                for param in child.parameters():
                    param.requires_grad = False
                j += 1
        


def load_pretrained_weights(model_dir, model):
    with open(model_dir, "rb") as fp:
        state_dict = pickle.load(fp)

    # Conv1
    model.conv1[0].weight.data.copy_(torch.from_numpy(state_dict["conv1_new"]["weights"].transpose((3,2,0,1))))
    model.conv1[0].bias.data.copy_(torch.from_numpy(state_dict["conv1_new"]["bias"]))

    model.conv1[1].weight.data.copy_(torch.from_numpy(state_dict["bn_conv1"]["scale"]))
    model.conv1[1].bias.data.copy_(torch.from_numpy(state_dict["bn_conv1"]["bias"]))
    model.conv1[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn_conv1"]["mean"]))
    model.conv1[1].running_var.data.copy_(torch.from_numpy(state_dict["bn_conv1"]["var"]))

    #######################################################################################
    # Res2a block
    # -- branch1 -> conv block
    model.res2a.conv_block[0].weight.data.copy_(torch.from_numpy(state_dict["res2a_branch1"]["weights"].transpose((3,2,0,1))))
    model.res2a.conv_block[0].bias.data.fill_(0.)

    model.res2a.conv_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn2a_branch1"]["scale"]))
    model.res2a.conv_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn2a_branch1"]["bias"]))
    model.res2a.conv_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn2a_branch1"]["mean"]))
    model.res2a.conv_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn2a_branch1"]["var"]))
    
    # --branch2 -> basic_block 
    model.res2a.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res2a_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res2a.basic_block[0].bias.data.fill_(0.)
    
    model.res2a.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn2a_branch2a"]["scale"]))
    model.res2a.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn2a_branch2a"]["bias"]))
    model.res2a.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn2a_branch2a"]["mean"]))
    model.res2a.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn2a_branch2a"]["var"]))

    model.res2a.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res2a_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res2a.basic_block[3].bias.data.fill_(0.)
    
    model.res2a.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn2a_branch2b"]["scale"]))
    model.res2a.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn2a_branch2b"]["bias"]))
    model.res2a.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn2a_branch2b"]["mean"]))
    model.res2a.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn2a_branch2b"]["var"]))

    model.res2a.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res2a_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res2a.basic_block[6].bias.data.fill_(0.)
    
    model.res2a.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn2a_branch2c"]["scale"]))
    model.res2a.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn2a_branch2c"]["bias"]))
    model.res2a.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn2a_branch2c"]["mean"]))
    model.res2a.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn2a_branch2c"]["var"]))

    # Res2b block
    # --branch2 -> basic_block 
    model.res2b.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res2b_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res2b.basic_block[0].bias.data.fill_(0.)
    
    model.res2b.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn2b_branch2a"]["scale"]))
    model.res2b.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn2b_branch2a"]["bias"]))
    model.res2b.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn2b_branch2a"]["mean"]))
    model.res2b.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn2b_branch2a"]["var"]))

    model.res2b.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res2b_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res2b.basic_block[3].bias.data.fill_(0.)
    
    model.res2b.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn2b_branch2b"]["scale"]))
    model.res2b.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn2b_branch2b"]["bias"]))
    model.res2b.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn2b_branch2b"]["mean"]))
    model.res2b.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn2b_branch2b"]["var"]))

    model.res2b.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res2b_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res2b.basic_block[6].bias.data.fill_(0.)
    
    model.res2b.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn2b_branch2c"]["scale"]))
    model.res2b.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn2b_branch2c"]["bias"]))
    model.res2b.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn2b_branch2c"]["mean"]))
    model.res2b.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn2b_branch2c"]["var"]))

    # Res2c block
    # --branch2 -> basic_block 
    model.res2c.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res2c_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res2c.basic_block[0].bias.data.fill_(0.)
    
    model.res2c.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn2c_branch2a"]["scale"]))
    model.res2c.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn2c_branch2a"]["bias"]))
    model.res2c.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn2c_branch2a"]["mean"]))
    model.res2c.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn2c_branch2a"]["var"]))

    model.res2c.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res2c_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res2c.basic_block[3].bias.data.fill_(0.)
    
    model.res2c.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn2c_branch2b"]["scale"]))
    model.res2c.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn2c_branch2b"]["bias"]))
    model.res2c.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn2c_branch2b"]["mean"]))
    model.res2c.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn2c_branch2b"]["var"]))

    model.res2c.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res2c_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res2c.basic_block[6].bias.data.fill_(0.)
    
    model.res2c.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn2c_branch2c"]["scale"]))
    model.res2c.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn2c_branch2c"]["bias"]))
    model.res2c.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn2c_branch2c"]["mean"]))
    model.res2c.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn2c_branch2c"]["var"]))

    #######################################################################################
    # Res3a block
    # -- branch1 -> conv block
    model.res3a.conv_block[0].weight.data.copy_(torch.from_numpy(state_dict["res3a_branch1"]["weights"].transpose((3,2,0,1))))
    model.res3a.conv_block[0].bias.data.fill_(0.)

    model.res3a.conv_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn3a_branch1"]["scale"]))
    model.res3a.conv_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn3a_branch1"]["bias"]))
    model.res3a.conv_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn3a_branch1"]["mean"]))
    model.res3a.conv_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn3a_branch1"]["var"]))
    
    # --branch2 -> basic_block 
    model.res3a.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res3a_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res3a.basic_block[0].bias.data.fill_(0.)
    
    model.res3a.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn3a_branch2a"]["scale"]))
    model.res3a.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn3a_branch2a"]["bias"]))
    model.res3a.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn3a_branch2a"]["mean"]))
    model.res3a.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn3a_branch2a"]["var"]))

    model.res3a.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res3a_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res3a.basic_block[3].bias.data.fill_(0.)
    
    model.res3a.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn3a_branch2b"]["scale"]))
    model.res3a.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn3a_branch2b"]["bias"]))
    model.res3a.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn3a_branch2b"]["mean"]))
    model.res3a.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn3a_branch2b"]["var"]))

    model.res3a.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res3a_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res3a.basic_block[6].bias.data.fill_(0.)
    
    model.res3a.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn3a_branch2c"]["scale"]))
    model.res3a.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn3a_branch2c"]["bias"]))
    model.res3a.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn3a_branch2c"]["mean"]))
    model.res3a.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn3a_branch2c"]["var"]))

    # Res3b block    
    # --branch2 -> basic_block 
    model.res3b.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res3b_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res3b.basic_block[0].bias.data.fill_(0.)
    
    model.res3b.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn3b_branch2a"]["scale"]))
    model.res3b.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn3b_branch2a"]["bias"]))
    model.res3b.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn3b_branch2a"]["mean"]))
    model.res3b.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn3b_branch2a"]["var"]))

    model.res3b.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res3b_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res3b.basic_block[3].bias.data.fill_(0.)
    
    model.res3b.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn3b_branch2b"]["scale"]))
    model.res3b.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn3b_branch2b"]["bias"]))
    model.res3b.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn3b_branch2b"]["mean"]))
    model.res3b.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn3b_branch2b"]["var"]))

    model.res3b.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res3b_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res3b.basic_block[6].bias.data.fill_(0.)
    
    model.res3b.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn3b_branch2c"]["scale"]))
    model.res3b.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn3b_branch2c"]["bias"]))
    model.res3b.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn3b_branch2c"]["mean"]))
    model.res3b.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn3b_branch2c"]["var"]))

    # Res3c block    
    # --branch2 -> basic_block 
    model.res3c.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res3c_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res3c.basic_block[0].bias.data.fill_(0.)
    
    model.res3c.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn3c_branch2a"]["scale"]))
    model.res3c.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn3c_branch2a"]["bias"]))
    model.res3c.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn3c_branch2a"]["mean"]))
    model.res3c.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn3c_branch2a"]["var"]))

    model.res3c.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res3c_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res3c.basic_block[3].bias.data.fill_(0.)
    
    model.res3c.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn3c_branch2b"]["scale"]))
    model.res3c.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn3c_branch2b"]["bias"]))
    model.res3c.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn3c_branch2b"]["mean"]))
    model.res3c.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn3c_branch2b"]["var"]))

    model.res3c.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res3c_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res3c.basic_block[6].bias.data.fill_(0.)
    
    model.res3c.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn3c_branch2c"]["scale"]))
    model.res3c.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn3c_branch2c"]["bias"]))
    model.res3c.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn3c_branch2c"]["mean"]))
    model.res3c.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn3c_branch2c"]["var"]))  

    ##########################################################################################
    # Res4a block
    # -- branch1 -> conv block
    model.res4a.conv_block[0].weight.data.copy_(torch.from_numpy(state_dict["res4a_branch1"]["weights"].transpose((3,2,0,1))))
    model.res4a.conv_block[0].bias.data.fill_(0.)

    model.res4a.conv_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn4a_branch1"]["scale"]))
    model.res4a.conv_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn4a_branch1"]["bias"]))
    model.res4a.conv_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4a_branch1"]["mean"]))
    model.res4a.conv_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4a_branch1"]["var"]))
    
    # --branch2 -> conv_block 
    model.res4a.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res4a_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res4a.basic_block[0].bias.data.fill_(0.)
    
    model.res4a.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn4a_branch2a"]["scale"]))
    model.res4a.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn4a_branch2a"]["bias"]))
    model.res4a.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4a_branch2a"]["mean"]))
    model.res4a.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4a_branch2a"]["var"]))

    model.res4a.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res4a_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res4a.basic_block[3].bias.data.fill_(0.)
    
    model.res4a.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn4a_branch2b"]["scale"]))
    model.res4a.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn4a_branch2b"]["bias"]))
    model.res4a.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn4a_branch2b"]["mean"]))
    model.res4a.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn4a_branch2b"]["var"]))

    model.res4a.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res4a_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res4a.basic_block[6].bias.data.fill_(0.)
    
    model.res4a.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn4a_branch2c"]["scale"]))
    model.res4a.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn4a_branch2c"]["bias"]))
    model.res4a.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn4a_branch2c"]["mean"]))
    model.res4a.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn4a_branch2c"]["var"]))

    # Res4b block
    # --branch2 -> conv_block 
    model.res4b.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res4b_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res4b.basic_block[0].bias.data.fill_(0.)
    
    model.res4b.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn4b_branch2a"]["scale"]))
    model.res4b.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn4b_branch2a"]["bias"]))
    model.res4b.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4b_branch2a"]["mean"]))
    model.res4b.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4b_branch2a"]["var"]))

    model.res4b.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res4b_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res4b.basic_block[3].bias.data.fill_(0.)
    
    model.res4b.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn4b_branch2b"]["scale"]))
    model.res4b.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn4b_branch2b"]["bias"]))
    model.res4b.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn4b_branch2b"]["mean"]))
    model.res4b.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn4b_branch2b"]["var"]))

    model.res4b.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res4b_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res4b.basic_block[6].bias.data.fill_(0.)
    
    model.res4b.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn4b_branch2c"]["scale"]))
    model.res4b.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn4b_branch2c"]["bias"]))
    model.res4b.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn4b_branch2c"]["mean"]))
    model.res4b.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn4b_branch2c"]["var"]))

    # Res4c block    
    # --branch2 -> basic_block 
    model.res4c.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res4c_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res4c.basic_block[0].bias.data.fill_(0.)
    
    model.res4c.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn4c_branch2a"]["scale"]))
    model.res4c.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn4c_branch2a"]["bias"]))
    model.res4c.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4c_branch2a"]["mean"]))
    model.res4c.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4c_branch2a"]["var"]))

    model.res4c.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res4c_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res4c.basic_block[3].bias.data.fill_(0.)
    
    model.res4c.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn4c_branch2b"]["scale"]))
    model.res4c.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn4c_branch2b"]["bias"]))
    model.res4c.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn4c_branch2b"]["mean"]))
    model.res4c.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn4c_branch2b"]["var"]))

    model.res4c.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res4c_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res4c.basic_block[6].bias.data.fill_(0.)
    
    model.res4c.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn4c_branch2c"]["scale"]))
    model.res4c.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn4c_branch2c"]["bias"]))
    model.res4c.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn4c_branch2c"]["mean"]))
    model.res4c.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn4c_branch2c"]["var"]))
    
    # Res4d block    
    # --branch2 -> basic_block 
    model.res4d.basic_block[0].weight.data.copy_(torch.from_numpy(state_dict["res4d_branch2a"]["weights"].transpose((3,2,0,1))))
    model.res4d.basic_block[0].bias.data.fill_(0.)
    
    model.res4d.basic_block[1].weight.data.copy_(torch.from_numpy(state_dict["bn4d_branch2a"]["scale"]))
    model.res4d.basic_block[1].bias.data.copy_(torch.from_numpy(state_dict["bn4d_branch2a"]["bias"]))
    model.res4d.basic_block[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4d_branch2a"]["mean"]))
    model.res4d.basic_block[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4d_branch2a"]["var"]))

    model.res4d.basic_block[3].weight.data.copy_(torch.from_numpy(state_dict["res4d_branch2b"]["weights"].transpose((3,2,0,1))))
    model.res4d.basic_block[3].bias.data.fill_(0.)
    
    model.res4d.basic_block[4].weight.data.copy_(torch.from_numpy(state_dict["bn4d_branch2b"]["scale"]))
    model.res4d.basic_block[4].bias.data.copy_(torch.from_numpy(state_dict["bn4d_branch2b"]["bias"]))
    model.res4d.basic_block[4].running_mean.data.copy_(torch.from_numpy(state_dict["bn4d_branch2b"]["mean"]))
    model.res4d.basic_block[4].running_var.data.copy_(torch.from_numpy(state_dict["bn4d_branch2b"]["var"]))

    model.res4d.basic_block[6].weight.data.copy_(torch.from_numpy(state_dict["res4d_branch2c"]["weights"].transpose((3,2,0,1))))
    model.res4d.basic_block[6].bias.data.fill_(0.)
    
    model.res4d.basic_block[7].weight.data.copy_(torch.from_numpy(state_dict["bn4d_branch2c"]["scale"]))
    model.res4d.basic_block[7].bias.data.copy_(torch.from_numpy(state_dict["bn4d_branch2c"]["bias"]))
    model.res4d.basic_block[7].running_mean.data.copy_(torch.from_numpy(state_dict["bn4d_branch2c"]["mean"]))
    model.res4d.basic_block[7].running_var.data.copy_(torch.from_numpy(state_dict["bn4d_branch2c"]["var"]))
    
    ###################################################################################################
    # Conv4e
    model.conv4e[0].weight.data.copy_(torch.from_numpy(state_dict["conv4e"]["weights"].transpose((3,2,0,1))))
    model.conv4e[0].bias.data.fill_(0.)
    model.conv4e[1].weight.data.copy_(torch.from_numpy(state_dict["bn4e"]["scale"]))
    model.conv4e[1].bias.data.copy_(torch.from_numpy(state_dict["bn4e"]["bias"]))
    model.conv4e[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4e"]["mean"]))
    model.conv4e[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4e"]["var"]))
    
    # Conv4f 
    model.conv4f[0].weight.data.copy_(torch.from_numpy(state_dict["conv4f"]["weights"].transpose((3,2,0,1))))
    model.conv4f[0].bias.data.fill_(0.)

    model.conv4f[1].weight.data.copy_(torch.from_numpy(state_dict["bn4f"]["scale"]))
    model.conv4f[1].bias.data.copy_(torch.from_numpy(state_dict["bn4f"]["bias"]))
    model.conv4f[1].running_mean.data.copy_(torch.from_numpy(state_dict["bn4f"]["mean"]))
    model.conv4f[1].running_var.data.copy_(torch.from_numpy(state_dict["bn4f"]["var"]))


mean_dict = \
{
"0": np.array([40.712, 79.706, 75.669, 75.358, 74.556, ]),
"1": np.array([40.712, 34.040, 53.132, ]),
"2": np.array([34.040, 29.417, ]),
"3": np.array([26.423, 29.417, ]),
"4": np.array([124.853, 26.423, ]),
"5": np.array([79.706, 53.132, 35.224, 23.676, ]),
"6": np.array([35.224, 23.270, ]),
"7": np.array([23.270, 22.036, ]),
"8": np.array([141.457, 22.036, ]),
"9": np.array([75.669, 23.676, 41.975, 18.504, ]),
"10": np.array([41.975, 26.329, ]),
"11": np.array([26.329, 24.504, ]),
"12": np.array([145.282, 24.504, ]),
"13": np.array([75.358, 18.504, 39.978, 18.243, ]),
"14": np.array([39.978, 23.513, ]),
"15": np.array([23.513, 22.647, ]),
"16": np.array([133.072, 22.647, ]),
"17": np.array([74.556, 18.243, 27.541, ]),
"18": np.array([27.541, 19.826, ]),
"19": np.array([19.826, 20.395, ]),
"20": np.array([127.492, 20.395, ]),
}

links_dict = \
{  "0":  [1, 5, 9, 13, 17],   # wrist: [T0, I0, M0, R0, L0]
   "1":  [0, 2, 5],           # T0:    [W,  T1]
   "2":  [1, 3],              # T1:    [T0, T2]
   "3":  [4, 2],              # T2:    [T1, T3]
   "4":  [0, 3],              # T3:    [W,  T2]
   "5":  [0, 1, 6, 9],   
   "6":  [5, 7], 
   "7":  [6, 8],
   "8":  [0, 7], 
   "9":  [0, 5, 10, 13],
   "10": [9, 11],
   "11": [10, 12],
   "12": [0, 11],
   "13": [0, 9, 14, 17], 
   "14": [13, 15], 
   "15": [14, 16], 
   "16": [0, 15],
   "17": [0, 13, 18],
   "18": [17, 19],
   "19": [18, 20],
   "20": [0, 19]
}
