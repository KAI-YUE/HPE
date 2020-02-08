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


def pre_process(image):
    """
    Convert a ndarray image ranging between [0, 1] to tensor (prepared for a model.)
    ----------------------------------------------------------
    Args,
        image:          ndarray (H x W x c), an image in rgb/grayscale format.
    Returns
        tensor_image:   tensor (1 x c x H x W)
    """
    # RGB format
    if (len(image.shape) == 3):
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        
        image = (image - mean) / std
        image = image.transpose(2,0,1)

        return torch.from_numpy(image).view(1, image.shape[0], image.shape[1], image.shape[2]).to(torch.float32)
    
    # Gray scale format
    else:
        return torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1]).to(torch.float32)


def pos_from_heatmap(Tensor_hm):
    """
    Get the pose array from the tensor heatmap (output of JOR network.)
    -------------------------------------------------------------------------------
    Args,  
        Tensor_hm,  tensor(N x C x h x w [32 x 32]), the heatmap tensors.  
    Returns,
        pos_arr:    ndarray([N x ] 21 x 2 ), each row corresponds to [col, row] (for plot purpose)
    """ 
    if len(Tensor_hm.shape) == 4:
        pos_arr = np.zeros((Tensor_hm.shape[0], 21, 2))
        for i in range(Tensor_hm.shape[0]):
            for j in range(21):
                index = torch.argmax(Tensor_hm[i,j])
                pos_arr[i,j,1] = int(index // Tensor_hm[i,j].shape[1] * 128/Tensor_hm[i,j].shape[0])
                pos_arr[i,j,0] = int(index % Tensor_hm[i,j].shape[1] * 128/Tensor_hm[i,j].shape[1])
    else:
        pos_arr = np.zeros((21, 2), dtype=int)
        for i in range(21):
            index = torch.argmax(Tensor_hm[i])
            pos_arr[i,1] = int(index // Tensor_hm[i].shape[0] * 128/Tensor_hm[i].shape[0])
            pos_arr[i,0] = int(index % Tensor_hm[i].shape[1] * 128/Tensor_hm[i].shape[1])

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
        gt_heatmap:   tensor (N x 1 x h x w), ground truth of the confidence map, to be upsampled. 
        output:       tensor (N x 1 x h x w), predicted confidence map, to be upsampled.
        epoch:        integer. The training epoch used for file name. 
    """

    for i in range(image.shape[0]):
        img = post_process(image[i, ...].cpu().detach())
        img = 255*img.transpose(1,2,0)

        heatmap = gt_heatmap[i, 0, ...].cpu().detach().numpy()
        heatmap = Heatmap(heatmap)

        pred = output[i, 0, ...].cpu().detach().numpy()
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]))
        pred = Heatmap(pred)

        alpha = 0.6
        output_image = np.hstack((alpha*img + (1-alpha)*heatmap, alpha*img + (1-alpha)*pred))

        cv2.imwrite(os.path.join(dir, "smaple_{}_{}.jpg".format(epoch, i)), output_image)


def project2plane(pos_3d):
    """
    Project the 3d position array to 2d plane. Camera parameters: f_x = 475.6 f_y = 475.62 x_0 = 311.125 y_0 = 245.965
    -------------------------------------------------------------------------
    Args,
        pos_3d:    ndarray(21 x 3), the 3d position array. 
    """
    f_x = 475.62
    f_y = 475.62
    x_0 = 311.125
    y_0 = 245.965

    num_parts = 21
    pos_arr = np.zeros((num_parts, 2))

    for i in range(num_parts):
        pos_arr[i, 0] = f_x / pos_3d[i, 2] * pos_3d[i, 0]
        pos_arr[i, 1] = f_y / pos_3d[i, 2] * pos_3d[i, 1]

    return pos_arr

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


def skin_mask(img):
    """
    Generate a skin mask based on the rule : Cr \in [139, 210] & Cb \in [77, 127]
    -----------------------------------------------------------------------------------
    Args,
        img:    an img in BGR format
    Retruns,
        mask:   a binary mask with skin pixels set to 1. (with 3 channels)
    """
    
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    mask = (YCrCb[...,2]>=77) * (YCrCb[...,2]<=127) * (YCrCb[...,1]>=139) * (YCrCb[...,1]<=210)         
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    return np.repeat(mask, 3, 2)