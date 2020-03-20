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
from utils.heatmap import Heatmap, Gaussian_heatmap
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
    plot_rows = 1
    plot_cols = 3
    num_parts = 21
    root_index = 9

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
                
                pos = 1000*a_set["3d_pos"] + a_set["root_pos"]
                gt_pos = pos.copy()
                pos -= pos[0]
                ROI = a_set["ROI"]
                img = a_set["cropped_img"]
                depth = a_set["cropped_depth_norm"]
                depth_with_img = np.dstack((depth, img)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None, ...].to(device)

                result = model(Img, torch.from_numpy(a_set["R_inv"].astype("float32")[None,...]).to(device))
                loss = L(result["pos"], torch.from_numpy(pos[1:].astype("float32")/1000).view(1,1,-1,3).to(device))
                fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 15))
                                
                # Plot the original image
                axs[0].set_axis_off()
                axs[0].set_title("Original")  
                axs[0].imshow((255*a_set["img"]).astype("uint8"))
                img = (255*a_set["cropped_img"]).astype("uint8")

                _3d_pos_arr_ = np.zeros((21,3))
                _3d_pos_arr_[1:] = 1000*result["pos"].cpu().detach().view(20,3).numpy().copy()
                _3d_pos_arr_ += gt_pos[0]

                _3d_error = np.mean( np.sqrt(np.sum( (_3d_pos_arr_ - gt_pos)**2, axis=1 )) )
                _2d_pos_arr_ = project2plane(_3d_pos_arr_)
                _2d_pos_arr_[:,0] = (_2d_pos_arr_[:,0] - ROI[2])*a_set["scale_factors"][1]
                _2d_pos_arr_[:,1] = (_2d_pos_arr_[:,1] - ROI[0])*a_set["scale_factors"][0]

                axs[1].set_axis_off()
                axs[1].set_title("3d Pred {:.2f}".format(_3d_error))
                plot_joint(img, _2d_pos_arr_, axs[1])

                # Plot the original link results
                ori_error = np.mean( np.sqrt(np.sum( (result["pos"].cpu().detach().view(20,3).numpy() - a_set["norm_3d_pos"][1:])**2, axis=1 )) )

                axs[2].set_axis_off()
                axs[2].set_title("Ground Truth {:.2f}".format(ori_error))  
                _2d_pos = a_set["2d_pos"]
                _2d_pos[:,0] = (_2d_pos[:,0] - ROI[2]) * a_set["scale_factors"][1]
                _2d_pos[:,1] = (_2d_pos[:,1] - ROI[0])* a_set["scale_factors"][0]
                plot_joint(img, _2d_pos, axs[2])
                
                sampled_in_folder += 1
                if (sampled_in_folder > config.samples_per_folder):
                    break
                
            already_sampled += sampled_in_folder
            if (already_sampled > config.test_samples):
                break
    
    networks_output = np.asarray(networks_output)
    np.save(os.path.join(config.test_output_dir, "DAE_1000_space.npy"), networks_output)


def Dexter_test(model_set, input_dir, output_dir, device="cuda"):
    """
    Test the joint model on EgoDexter dataset. First the HLo model localize the hand image, then 
    the PRe model regresses the joint position.
    --------------------------------------------------------------------------------
    Args,
        model_set,   the set of model including HLo(hand localization), JLo(joint localization), PRe (Position Regression)
        input_dir,   the input directory of the test set (already transformed.) The grountruth is unknown.  
    """
    alpha = 0.7
    invalid_depth = 0
    depth_max = 1000
    scale_factors = np.array([0.,0.])
    config = loadConfig()
    
    cropped_size = tuple(config.cropped_size)
    already_sampled = 0

    # Load networks model
    HLo = model_set["HLo"].eval()
    JLo = model_set["JLo"].eval()
    VPE = model_set["VPE"].eval()
    PRe = model_set["PRe"].eval()
    
    # Plot rows x cols to show results
    plot_rows = 2
    plot_cols = 4
    accumulated_3d_error = 0
    accumulated_3d_error_precise = 0
    error_th = 60
    error_list = []
    error_precise = []
    fingertip_indices = [4,8,12,16,20]

    for root, dirs, files in os.walk(input_dir):
        
        new_dir = os.path.join(root.replace(input_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            indices = np.arange(len(files))
            # random.shuffle(indices)

            for i in range(indices.shape[0]):
                f = files[indices[i]]
                with open(os.path.join(root, f), "rb") as fp:
                    a_set = pickle.load(fp)
                
                if not "palm_pos" in a_set:
                    continue

                img = a_set["img"]
                depth = a_set["depth"]
                depth_norm = a_set["depth_norm"]
                depth_with_img = np.dstack((depth_norm, img)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None, ...].to(device)
                
                root_heatmap = HLo(Img)
                center = center_from_heatmap(root_heatmap.squeeze())

                ROI_with_mean_depth = ROI_Hand(img, depth, center, invalid_depth)
                ROI = ROI_with_mean_depth["ROI"]
                mean_depth = ROI_with_mean_depth["mean_depth"]

                scale_factors[0] = cropped_size[1]/(ROI[1]-ROI[0])
                scale_factors[1] = cropped_size[0]/(ROI[3]-ROI[2])

                root_heatmap_numpy = root_heatmap.squeeze().detach().cpu().numpy()
                img_display = (255*img).astype("uint8")
                heatmap = Heatmap(root_heatmap_numpy)
                composite = (alpha*img_display + (1-alpha)*heatmap[...,::-1]).astype("uint8")
                
                # Plot the result 
                fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(30, 15))
                axs[0,0].set_axis_off()
                axs[0,0].set_title("Original image")
                axs[0,0].imshow(img_display)
                
                axs[0,1].set_axis_off()
                axs[0,1].set_title("Predicted {}".format(center))
                axs[0,1].imshow(composite)

                # Plot the cropped hand
                cropped_hand = img_display[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                cropped_hand = cv2.resize(cropped_hand, cropped_size)
                cropped_depth = depth[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                cropped_depth = cv2.resize(cropped_depth, cropped_size, interpolation=cv2.INTER_NEAREST)
                cropped_depth_norm = np.where(cropped_depth==invalid_depth, depth_max+mean_depth, cropped_depth)
                cropped_depth_norm = np.where(cropped_depth_norm>depth_max, depth_max+mean_depth, cropped_depth_norm)
                cropped_depth_norm = (cropped_depth_norm - mean_depth)/depth_max

                axs[0,2].set_axis_off()
                axs[0,2].set_title("Cropped")
                axs[0,2].imshow(cropped_hand)

                # Regress the viewpoint and joint position
                depth_with_img = np.dstack((cropped_depth_norm, cropped_hand/255)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None,...].to(device)

                heatmaps = JLo(Img)
                heatmaps_numpy = heatmaps.detach().cpu().squeeze().numpy() 
                pred_2d_pos = naive_pos_from_heatmap(heatmaps_numpy)
                pred_2d_pos[:,0] = pred_2d_pos[:,0]/scale_factors[1] + ROI[2] 
                pred_2d_pos[:,1] = pred_2d_pos[:,1]/scale_factors[0] + ROI[0]

                pos0 = back_project(pred_2d_pos[0], depth)

                vpe_output = VPE(Img)
                pre_output = PRe(Img, vpe_output["R_inv"])
                pred_3d_pos = pre_output["pos"]
                pred_3d_pos = pred_3d_pos.squeeze()
                pred_3d_pos_numpy = 1000*pred_3d_pos.detach().cpu().numpy()
                pred_3d_pos_numpy = np.vstack((np.zeros(3), pred_3d_pos_numpy))
                pred_3d_pos_numpy = pred_3d_pos_numpy + pos0 
                
                # plot the palm center heatmap
                heatmap = Heatmap(heatmaps_numpy[0])
                composite = (alpha*cropped_hand + (1-alpha)*heatmap[...,::-1]).astype("uint8")
                axs[0,3].set_axis_off()
                axs[0,3].set_title("Palm Center")
                axs[0,3].imshow(composite)
                
                # plot the precise plam center heatmap
                pos0_precise = a_set["palm_pos"]
                proj_pos0_precise = project2plane(pos0_precise.reshape((1,3)))
                proj_pos0_precise[0,0] = (proj_pos0_precise[0,0] - ROI[2])*scale_factors[1]
                proj_pos0_precise[0,1] = (proj_pos0_precise[0,1] - ROI[0])*scale_factors[0] 
                heatmap_set = Gaussian_heatmap(cropped_hand, proj_pos0_precise.flatten())
                axs[1,0].set_axis_off()
                axs[1,0].set_title("Palm Center")
                axs[1,0].imshow(heatmap_set["composite"].astype("uint8"))              

                # plot the original annotations
                _2d_pos = a_set["2d_pos"]
                _3d_pos = a_set["3d_pos"]
                R_inv_precise = a_set["R_inv"].astype("float32")
                R_inv_precise = torch.from_numpy(R_inv_precise).view(1,3,3)
                R_inv_precise = R_inv_precise.to(device)

                # transform the pos array to the cropped image space
                _2d_pos_plot = _2d_pos.copy()
                _2d_pos_plot[:, 0] = (_2d_pos_plot[:, 0] - ROI[2])*scale_factors[1]
                _2d_pos_plot[:, 1] = (_2d_pos_plot[:, 1] - ROI[0])*scale_factors[0]

                axs[1,1].set_axis_off()
                axs[1,1].set_title("Keypoints GT")
                axs[1,1].imshow(cropped_hand)
                for j in range(_2d_pos_plot.shape[0]):
                    axs[1,1].scatter(_2d_pos[j,0], _2d_pos[j,1])
                
                # Plot the 3-D links results
                error = np.mean(np.sqrt(np.sum((pred_3d_pos_numpy[fingertip_indices]-_3d_pos)**2, axis=-1)))
                accumulated_3d_error += error
                error_list.append(error)

                proj_pred_3d_pos = project2plane(pred_3d_pos_numpy)
                proj_pred_3d_pos_plot = proj_pred_3d_pos
                proj_pred_3d_pos_plot[:,0] = (proj_pred_3d_pos_plot[:,0] - ROI[2]) * scale_factors[1]
                proj_pred_3d_pos_plot[:,1] = (proj_pred_3d_pos_plot[:,1] - ROI[0]) * scale_factors[0]

                axs[1,2].set_axis_off()
                axs[1,2].set_title("3D Links {:.3f}".format(error))
                plot_joint(cropped_hand, proj_pred_3d_pos, axs[1,2])
                
                # Predict again with the "precise" transformation
                pre_output = PRe(Img, R_inv_precise)
                pred_3d_pos = pre_output["pos"]
                pred_3d_pos = pred_3d_pos.squeeze()
                pred_3d_pos_numpy = 1000*pred_3d_pos.detach().cpu().numpy()
                pred_3d_pos_numpy = np.vstack((np.zeros(3), pred_3d_pos_numpy))
                pred_3d_pos_numpy = pred_3d_pos_numpy + pos0_precise 

                error = np.mean(np.sqrt(np.sum((pred_3d_pos_numpy[fingertip_indices]-_3d_pos)**2, axis=-1)))
                if error > error_th:
                    continue
                accumulated_3d_error_precise += error
                error_precise.append(error)

                proj_pred_3d_pos = project2plane(pred_3d_pos_numpy)
                proj_pred_3d_pos_plot = proj_pred_3d_pos
                proj_pred_3d_pos_plot[:,0] = (proj_pred_3d_pos_plot[:,0] - ROI[2]) * scale_factors[1]
                proj_pred_3d_pos_plot[:,1] = (proj_pred_3d_pos_plot[:,1] - ROI[0]) * scale_factors[0]

                axs[1,3].set_axis_off()
                axs[1,3].set_title("3D Links {:.3f}".format(error))
                plot_joint(cropped_hand, proj_pred_3d_pos, axs[1,3])

                fig.savefig(os.path.join(new_dir, f[:5] + ".jpg"))
                plt.close(fig)

                sampled_in_folder += 1

                if (sampled_in_folder > config.samples_per_folder):
                    break
        
            already_sampled += sampled_in_folder
            if (already_sampled > config.test_samples):
                break
    
    averaged_error = accumulated_3d_error/already_sampled 
    averaged_error_precise = accumulated_3d_error_precise/already_sampled
    np.savetxt(os.path.join(config.test_output_dir, "error_PCA20.txt"), np.asarray([averaged_error, averaged_error_precise]))
    np.savetxt(os.path.join(config.test_output_dir, "error_arr.txt"), np.asarray(error_list))
    np.savetxt(os.path.join(config.test_output_dir, "error_precise.txt"), np.asarray(error_precise))

        
