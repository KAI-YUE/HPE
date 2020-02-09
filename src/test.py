# Python Libraries
import os
import pickle
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Pytorch Libraries
import torch
from torch.utils.data import DataLoader

# My Libraries
from src.loadConfig import loadConfig
from utils.heatmap import Heatmap
from utils.hand_region import ROI_Hand
from utils.plot_pos import plot_joint
from src.loss import HLoCriterion, PReCriterion
from src.dataset import HLoDataset, PReDataset
from utils.tools import *

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
        
        already_sampled = 0
        L = HLoCriterion()

        for root, dirs, files in os.walk(config.test_dir):

            new_dir = os.path.join(root.replace(config.test_dir, output_dir))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            if (files != []):
                sampled_in_folder = 0
                for f in files:
                    if ".dat" in f:
                        with open(os.path.join(root, f), "rb") as fp:
                            a_set = pickle.load(fp)

                        img = a_set["img"]
                        hm = a_set["W_ori"]

                        Tensor_img = pre_process(img).to(device)
                        Tensor_hm = pre_process(hm).to(device)

                        result = model(Tensor_img)
                        loss = L(result, Tensor_hm)
                        
                        result = np.squeeze(result.cpu().detach().numpy())
                        result = cv2.resize(result, tuple(config.input_size))
                        
                        heatmap = Heatmap(result)
                        composite = 255 * alpha * img + (1 - alpha) * heatmap
                        cv2.imwrite(os.path.join(new_dir, f[:8] + "_{:.2f}.jpg".format(loss.cpu().detach())), composite)

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
        
    already_sampled = 0
    L = PReCriterion()

    # Plot rows x cols to show results
    plot_rows = 4
    plot_cols = 6

    for root, dirs, files in os.walk(config.test_dir):

        new_dir = os.path.join(root.replace(config.test_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            for f in files:
                if ".dat" in f:
                    with open(os.path.join(root, f), "rb") as fp:
                        a_set = pickle.load(fp)
                        
                    img = a_set["hand"]

                    Tensor_img = pre_process(img).to(device)
                    Tensor_hm = torch.from_numpy(a_set['heatmaps']).to(torch.float32).to(device)
                    Tensor_hm = Tensor_hm[None, ...]
                    Tensor_pos = torch.from_numpy(a_set['pos_arr']).view(1,21,3,1).to(torch.float32).to(device)

                    result = model(Tensor_img)
                    loss, _ = L(result[0], result[1], Tensor_hm, Tensor_pos)
                    pos_arr = pos_from_heatmap(result[0].squeeze())

                    result = np.squeeze(result[0].cpu().detach().numpy())
                    result = result.transpose((1,2,0))

                    # Plot the result 
                    img = (255*img).astype("uint8")
                    fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 15))
                    counter = 0
                    for r in range(plot_rows):
                        for c in range(plot_cols):
                            hm = cv2.resize(result[...,counter], tuple(config.cropped_size))
                            heatmap = Heatmap(hm)

                            composite = (alpha*img + (1-alpha)*heatmap[...,::-1]).astype(np.uint8)
                            axs[r, c].set_axis_off()
                            axs[r, c].set_title(Part_Namelist[counter])
                            axs[r, c].imshow(composite)
                            
                            counter += 1
                            if counter == 21:
                                break

                    # plot the original hand image
                    axs[-1, 3].set_axis_off()
                    axs[-1, 3].imshow(img)
                    
                    # Plot the 2-D links results
                    axs[-1, 4].set_axis_off()  
                    plot_joint(img, pos_arr, axs[-1, 4])
                    
                    # Plot the original image
                    axs[-1, 5].set_axis_off()  
                    axs[-1, 5].imshow((255*a_set["img"]).astype("uint8"))

                    fig.savefig(os.path.join(new_dir, f[:8] + "_{:.2f}.jpg".format(loss.cpu().detach())))
                    np.save(os.path.join(new_dir, f[:8] + "hm.npy"), result)

                    plt.close(fig)
                    sampled_in_folder += 1
                    if (sampled_in_folder > config.samples_per_folder):
                        break
                    
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

    # Plot rows x cols to show results
    plot_rows = 2
    plot_cols = 3

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


def Ego_test(HLo, PRe, input_dir, output_dir, device="cuda"):
    """
    Test the joint model on EgoDexter dataset. First the HLo model localize the hand image, then 
    the PRe model regresses the joint position.
    --------------------------------------------------------------------------------
    Args,
        HLo,       the hand localization model.
        PRe,       the joint regression model.
        input_dir, the input directory of the test set (already transformed.) The grountruth is unknown.  
    """
    alpha = 0.6
    config = loadConfig()

    for root, dirs, files in os.walk(input_dir):
        
        new_dir = os.path.join(root.replace(input_dir, output_dir))
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
        
