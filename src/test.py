# Python Libraries
import os
import pickle
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Pytorch Libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# My Libraries
from src.loadConfig import loadConfig
from utils.heatmap import Heatmap
from utils.hand_region import ROI_Hand
from utils.plot_pos import plot_joint
from src.loss import HLoCriterion, PReCriterion
from src.dataset import HLoDataset, PReDataset
from utils.tools import *

f_x = 475.62
f_y = 475.62
x_0 = 311.125
y_0 = 245.965

Part_Namelist = ["W", 
"T0", "T1", "T2", "T3", 
"I0", "I1", "I2", "I3",
"M0", "M1", "M2", "M3",
"R0", "R1", "R2", "R3",
"L0", "L1", "L2", "L3"]

def HLo_test(model, output_dir, device="cuda", mode=0):
    """
    Test the hand localization model.
    -----------------------------------------------
    Args:
        output_dir,    the directory of the output results
        mode,          0: test on the EgoDexter (without quantitative loss)
                       1: test on the synthHands 
    """
    alpha = 0.6
    config = loadConfig()

    if mode == 0:

        already_sampled = 0
        
        for root, dirs, files in os.walk(config.test_dir):
            
            new_dir = os.path.join(root.replace(config.test_dir, output_dir))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            if (files != [] and "color_on_depth" in root):
                sampled_in_folder = 0
                indices = np.arange(len(files))
                # random.shuffle(indices)

                for i in range(config.test_samples):
                    f = files[indices[i]]
                    img = cv2.imread(os.path.join(root, f))[:,:,::-1]
                    img = cv2.resize(img, tuple(config.input_size))
                    
                    Tensor_img = pre_process(img/255).to(device)
                    result = model(Tensor_img)

                    result = np.squeeze(result.cpu().detach().numpy())
                    result = cv2.resize(result, tuple(config.input_size))
                    
                    heatmap = Heatmap(result)

                    composite = alpha * img + (1 - alpha) * heatmap
                    cv2.imwrite(os.path.join(new_dir, f.replace("color_on_the_depth", "composite")), composite)

                    sampled_in_folder += 1

                    if (sampled_in_folder > config.samples_per_folder):
                        break
                
                already_sampled += sampled_in_folder
                if (sampled_in_folder > config.test_samples):
                    break

    elif mode == 1:
        root_index = 9
        already_sampled = 0

        for root, dirs, files in os.walk(config.test_dir):

            new_dir = os.path.join(root.replace(config.test_dir, output_dir))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            if (files != []):
                sampled_in_folder = 0
                for f in files:
                    with open(os.path.join(root, f), "rb") as fp:
                        a_set = pickle.load(fp)

                    img = a_set["img"]
                    depth = a_set["depth_norm"]
                    _2d_pos = a_set["2d_pos"]

                    Tensor_img = torch.from_numpy(np.dstack((depth,img)).transpose((2,0,1))).to(torch.float32)
                    Tensor_img = Tensor_img[None,...].to(device)

                    result = model(Tensor_img)
                    center = center_from_heatmap(result.squeeze())
                    result = result.cpu().detach().squeeze().numpy()
                    
                    error = np.sqrt(np.sum((center - _2d_pos[root_index])**2))

                    heatmap = Heatmap(result)
                    composite = 255 * alpha * img + (1 - alpha) * heatmap
                    cv2.imwrite(os.path.join(new_dir, f[:8] + "_{:.2f}.jpg".format(error)), composite)

                    sampled_in_folder += 1

                    if (sampled_in_folder > config.samples_per_folder):
                            break
                
                already_sampled += sampled_in_folder
                if (sampled_in_folder > config.test_samples):
                    break


def PRe_test(model, output_dir, device="cuda"):
    """
    Test the joint regression model.
    -----------------------------------------------
    Args:
        output_dir,    the directory of the output results
        (Comment: Since Egodexter dataset has no grondtruth, we can only test on SynthHand dataset.)
    """
    alpha = 0.6
    config = loadConfig()
    cropped_size = tuple(config.cropped_size)

    already_sampled = 0

    # Plot rows x cols to show results
    plot_rows = 4
    plot_cols = 7
    num_parts = 21
    root_index = 9
    max_depth = 1000
    pos_scale = 1000

    L = PReCriterion()
    
    for root, dirs, files in os.walk(config.test_dir):

        new_dir = os.path.join(root.replace(config.test_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            for f in files:
                with open(os.path.join(root, f), "rb") as fp:
                    a_set = pickle.load(fp)
                
                img = a_set["cropped_img"]
                depth = a_set["cropped_depth"]
                ROI = a_set["ROI"]
                _3d_pos = 1000*a_set["3d_pos"] + a_set["root_pos"]
                _2d_pos = a_set["2d_pos"]
                
                data = np.dstack((depth, img)).astype('float32')
                data = torch.from_numpy(data.transpose((2,0,1))).to(torch.float32)
                result = model(data[None,...].to(device))

                hms = result['hms'].cpu().detach().squeeze().numpy()
                hms = cv2.resize(hms.transpose((1,2,0)), tuple(config.cropped_size)).transpose((2,0,1))
                pos = pos_scale*result['pos'].cpu().detach().squeeze().numpy()
                index = 0
                
                fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 15))
                hm_index = 0
                for r in range(1, plot_rows):
                    for c in range(plot_cols):
                        hm = hms[hm_index]
                        heatmap = Heatmap(hm)

                        composite = (alpha*255*img + (1-alpha)*heatmap[...,::-1]).astype(np.uint8)
                        axs[r, c].set_axis_off()
                        axs[r, c].set_title(Part_Namelist[hm_index])
                        axs[r, c].imshow(composite)
                        
                        hm_index += 1
                        if hm_index == num_parts:
                            break
                
                # Plot the original image
                axs[0,0].set_axis_off()
                axs[0,0].imshow((255*a_set["img"]).astype("uint8"))
                
                # Plot the cropped image
                img = (255*img).astype("uint8")
                axs[0,1].set_axis_off()
                axs[0,1].imshow(img)
                
                # Plot 2d link result
                pred_2dpos = naive_pos_from_heatmaps(hms[:-1])
                axs[0,2].set_axis_off()
                plot_joint(img, pred_2dpos, axs[0,2])
                
                pred_2dpos[:,0] = pred_2dpos[:,0]/a_set["scale_factors"][1] + ROI[2]  
                pred_2dpos[:,1] = pred_2dpos[:,1]/a_set["scale_factors"][0] + ROI[0]
                _2d_error = np.mean(np.sqrt(np.sum((pred_2dpos-_2d_pos)**2, axis=1)))
                axs[0,2].set_title("2d error {:.2f}".format(_2d_error))
                
                # Plot 3d projected result
                depth = a_set["depth"]
                root_pos = back_project(pred_2dpos[root_index], depth[pred_2dpos[root_index, 1], pred_2dpos[root_index, 0]])        
                pos += root_pos
                proj_2dpos = project2plane(pos)
                proj_2dpos[:,0] = (proj_2dpos[:,0] - ROI[2]) * a_set["scale_factors"][1]
                proj_2dpos[:,1] = (proj_2dpos[:,1] - ROI[0]) * a_set["scale_factors"][0]
                _3d_error = np.mean(np.sqrt(np.sum((pos-_3d_pos)**2, axis=1)))

                axs[0,3].set_axis_off()
                plot_joint(img, proj_2dpos, axs[0,3])
                axs[0,3].set_title("3d error {:.2f}".format(_3d_error))

                # Plot the original link result
                _2d_pos[:,0] = (_2d_pos[:,0] - ROI[2])*a_set["scale_factors"][1]
                _2d_pos[:,1] = (_2d_pos[:,1] - ROI[0])*a_set["scale_factors"][0]
                axs[0,4].set_axis_off()
                axs[0,4].set_title("Ori")
                plot_joint(img, _2d_pos, axs[0,4])

                plt.savefig(os.path.join(new_dir, f[:-4]+"_pos.png"))
                plt.close(fig)

            already_sampled += sampled_in_folder
            if (sampled_in_folder > config.test_samples):
                break


def Synth_test(HLo, PRe, input_dir, output_dir, device="cuda"):
    """
    Test the joint model on SynthHand dataset. First the HLo model localize the hand image, then 
    the PRe model regresses the joint position.
    --------------------------------------------------------------------------------
    Args,
        HLo,       the hand localization model.
        PRe,       the position regression model.
        input_dir, the input directory of the test set (already transformed.) The grountruth is known.  
    """

    alpha = 0.7
    config = loadConfig()

    already_sampled = 0

    for root, dirs, files in os.walk(input_dir):
        
        new_dir = os.path.join(root.replace(input_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            for f in files:
                if ".dat" in f:
                    with open(os.path.join(root, f), "rb") as fp:
                        a_set = pickle.load(fp)

                    # Localize the wrist
                    img = a_set["img"]
                    depth = a_set["depth"]
                    Tensor_img = pre_process(img).to(device)

                    result = HLo(Tensor_img)
                    center = center_from_heatmap(result.squeeze())

                    # Back projection with th predicted center
                    d = depth[center[1], center[0]]
                    x_3d = (center[0] - x_0) * d / f_x
                    y_3d = (center[1] - y_0) * d / f_y

                    hm = result.squeeze().detach().cpu().numpy()
                    hm = cv2.resize(hm, tuple(config.input_size))
                    img = (255*img).astype("uint8")
                    heatmap = Heatmap(hm)
                    composite = alpha*img + (1-alpha)*heatmap[...,::-1]
                    
                    # Plot the result 
                    fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(15, 10))
                    axs[0,0].set_axis_off()
                    axs[0,0].set_title("Original image")
                    axs[0,0].imshow(img)
                    
                    axs[1,0].set_axis_off()
                    axs[1,0].set_title("Predicted {}".format(center))
                    axs[1,0].imshow(composite.astype("uint8"))
                    
                    # Load the pos array
                    f_ = f.replace(".dat", "_joint_pos.txt")
                    pos_arr = np.loadtxt(os.path.join(root, f_))

                    # Plot the cropped hand
                    ROI = ROI_Hand(img, depth, center)
                    cropped_hand = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                    axs[0,1].set_axis_off()
                    axs[0,1].set_title("Cropped hm")
                    axs[0,1].imshow(cropped_hand)
                    
                    # modify the pos array.
                    pos_arr[21:, 0] -= ROI[2]
                    pos_arr[21:, 1] -= ROI[0]

                    # Regress the joint position
                    Tensor_img = pre_process(cropped_hand).to(device)

                    result, interm = PRe(Tensor_img)
                    pred_pos = pos_from_heatmap(result[0].squeeze())

                    pos_3d = result[1].cpu().detach().squeeze().numpy()
                    pos_3d += np.array(x_0, y_0, d)

                    # plot the original 2-D links
                    axs[1, 1].set_axis_off()
                    axs[1, 1].set_title("Links GT")
                    plot_joint(cropped_hand, pos_arr[21:, :2], axs[0, 2])
                    
                    # Plot the 2-D links results
                    axs[0, 2].set_axis_off()
                    axs[0, 2].set_title("2D Links ")  
                    plot_joint(cropped_hand, pred_pos, axs[0, 2])
                    
                    # Plot the 3-D links results
                    axs[1, 2].set_axis_off()
                    axs[1, 2].set_title("3D Links")
                    plot_joint(cropped_hand, pred_pos, axs[1, 2])

                    error = np.sqrt(np.sum((pos_arr[21:, :2]-pred_pos)**2))
                    
                    fig.savefig(os.path.join(new_dir, f[:8] + "_{:.2f}.jpg".format(error)))
                    plt.close(fig)

                    sampled_in_folder += 1

                    if (sampled_in_folder > config.samples_per_folder):
                        break
            
                already_sampled += sampled_in_folder
                if (sampled_in_folder > config.test_samples):
                    break


def Dexter_test(HLo, PRe, input_dir, output_dir, device="cuda"):
    """
    Test the joint model on EgoDexter dataset. First the HLo model localize the hand image, then 
    the PRe model regresses the joint position.
    --------------------------------------------------------------------------------
    Args,
        HLo,       the hand localization model.
        PRe,       the joint regression model.
        input_dir, the input directory of the test set (already transformed.) The grountruth is unknown.  
    """
    alpha = 0.7
    config = loadConfig()
    
    already_sampled = 0
    
    # Plot rows x cols to show results
    plot_rows = 4
    plot_cols = 7
    num_parts = 21

    for root, dirs, files in os.walk(input_dir):
        
        new_dir = os.path.join(root.replace(input_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            indices = np.arange(len(files))
            # random.shuffle(indices)

            for i in range(config.test_samples):
                f = files[indices[i]]
                with open(os.path.join(root, f), "rb") as fp:
                    a_set = pickle.load(fp)
                
                img = a_set["img"]
                depth = a_set["depth"]
                Tensor_img = pre_process(img).to(device)
                result = HLo(Tensor_img)
                
                center = center_from_heatmap(result.squeeze())

                # Back projection with th predicted center
                ##

                hm = result.squeeze().detach().cpu().numpy()
                img = (255*img).astype("uint8")
                heatmap = Heatmap(hm)
                composite = (alpha*img + (1-alpha)*heatmap[...,::-1]).astype("uint8")
                
                # Plot the result 
                fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(30, 15))
                axs[0,0].set_axis_off()
                axs[0,0].set_title("Original image")
                axs[0,0].imshow(img)
                
                axs[0,1].set_axis_off()
                axs[0,1].set_title("Predicted {}".format(center))
                axs[0,1].imshow(composite)
                
                # Load the pos array
                _2d_pos = a_set["2d_pos"]

                # Plot the cropped hand
                ROI = ROI_Hand(img/255, depth, center)
                cropped_hand = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                cropped_hand = cv2.resize(cropped_hand, tuple(config.cropped_size))

                axs[0,2].set_axis_off()
                axs[0,2].set_title("Cropped")
                axs[0,2].imshow(cropped_hand)
                
                # modify the pos array.
                _2d_pos[:, 0] -= ROI[2]
                _2d_pos[:, 1] -= ROI[0]

                # Regress the joint position
                Tensor_img = pre_process(cropped_hand).to(device)

                result = PRe(Tensor_img)
                hms = result[0].squeeze().cpu().detach().numpy()
                pred_pos = naive_pos_from_heatmap(hms)

                # pos_3d = result[1].cpu().detach().squeeze().numpy()
                # pos_3d += np.array(x_0, y_0, d)

                # plot the original annotations
                axs[0, 3].set_axis_off()
                axs[0, 3].set_title("Keypoints GT")
                axs[0, 3].imshow(cropped_hand)
                for j in range(_2d_pos.shape[0]):
                    axs[0, 3].scatter(_2d_pos[j,0], _2d_pos[j,1])
                
                # Plot the 2-D links results
                fingertip_indices = [4, 8, 12, 16, 20]
                error = np.mean(np.sqrt(np.sum((_2d_pos-pred_pos[fingertip_indices])**2, axis=1)))
                axs[0, 4].set_axis_off()
                axs[0, 4].set_title("2D Links {}".format(error))  
                plot_joint(cropped_hand, pred_pos, axs[0, 4])
                
                # Plot the 3-D links results
                # axs[0, 5].set_axis_off()
                # axs[0, 5].set_title("3D Links")
                # plot_joint(cropped_hand, pred_pos, axs[1, 2])

                # Plot the heatmaps of different parts
                counter = 0
                for r in range(1, plot_rows):
                    for c in range(plot_cols):
                        hm = cv2.resize(hms[counter], tuple(config.cropped_size))
                        heatmap = Heatmap(hm)

                        composite = (alpha*cropped_hand + (1-alpha)*heatmap[...,::-1]).astype(np.uint8)
                        axs[r, c].set_axis_off()
                        axs[r, c].set_title(Part_Namelist[counter])
                        axs[r, c].imshow(composite)
                        
                        counter += 1
                        if counter == num_parts:
                            break
                
                fig.savefig(os.path.join(new_dir, f[:8] + ".jpg"))
                plt.close(fig)

                sampled_in_folder += 1

                if (sampled_in_folder > config.samples_per_folder):
                    break
        
            already_sampled += sampled_in_folder
            if (sampled_in_folder > config.test_samples):
                break


        
