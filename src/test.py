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

    for root, dirs, files in os.walk(config.test_dir):

        new_dir = os.path.join(root.replace(config.test_dir, output_dir))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        if (files != []):
            sampled_in_folder = 0
            for f in files:
                with open(os.path.join(root, f), "rb") as fp:
                    a_set = pickle.load(fp)

                ROI = a_set["ROI"]
                img = a_set["cropped_img"]
                depth = a_set["cropped_depth"]
                depth_with_img = np.dstack((depth, img)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None, ...].to(device)

                root_pos = a_set["root_pos"]
                pos = a_set["3d_pos"]
                gt_pos = 1000*pos + root_pos

                Tensor_hms, Tensor_pos = model(Img)

                hms = Tensor_hms.cpu().detach().numpy().squeeze()

                # Plot the result 
                img = (255*img).astype("uint8")
                fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 15))
                counter = 0
                break_flag = False
                for r in range(1,plot_rows):
                    for c in range(plot_cols):
                        hm = cv2.resize(hms[counter], tuple(config.cropped_size))
                        heatmap = Heatmap(hm)

                        composite = (alpha*img + (1-alpha)*heatmap[...,::-1]).astype(np.uint8)
                        axs[r, c].set_axis_off()
                        axs[r, c].set_title(Part_Namelist[counter])
                        axs[r, c].imshow(composite)
                        
                        counter += 1
                        if counter == num_parts:
                            break_flag = True
                            break

                    if (break_flag == True):
                        break

                # # plot the original hand image
                # axs[-1, 3].set_axis_off()
                # axs[-1, 3].imshow(img)
                                
                # Plot the original image
                axs[0,1].set_axis_off()
                axs[0,1].set_title("Original")  
                axs[0,1].imshow((255*a_set["img"]).astype("uint8"))
                
                # Plot the link results with naive method
                _2d_pos_arr_ = naive_pos_from_heatmap(hms)
                axs[0,4].set_axis_off()
                plot_joint(img, _2d_pos_arr_, axs[0,4])

                _2d_pos_arr_[:,0] = _2d_pos_arr_[:,0]/a_set["scale_factors"][1] + ROI[2]
                _2d_pos_arr_[:,1] = _2d_pos_arr_[:,1]/a_set["scale_factors"][1] + ROI[0]
                _2d_error = np.mean(np.sqrt(np.sum((_2d_pos_arr_-a_set["2d_pos"])**2, axis=1)))
                axs[0,4].set_title("2d Pred {:.02f}".format(_2d_error))

                # Calculate the 3d pos distance and plot the projection of 3d pos
                pred_root_pos = _2d_pos_arr_[root_index].astype("int")
                pred_3d_root_pos = back_project(pred_root_pos, a_set["depth"])

                _3d_pos_arr_ = 1000*Tensor_pos.cpu().detach().numpy().squeeze() + a_set["root_pos"]
                _3d_error = np.mean( np.sqrt(np.sum( (_3d_pos_arr_-gt_pos)**2, axis=1 )) )
                _3d_pos_arr_ = project2plane(_3d_pos_arr_)
                _3d_pos_arr_[:,0] = (_3d_pos_arr_[:,0] - ROI[2])*a_set["scale_factors"][1]
                _3d_pos_arr_[:,1] = (_3d_pos_arr_[:,1] - ROI[0])*a_set["scale_factors"][0]

                axs[0,5].set_axis_off()
                axs[0,5].set_title("3d Pred {:.2f}".format(_3d_error))
                plot_joint(img, _3d_pos_arr_, axs[0,5])

                # Plot the original link results
                ori_error = 1000* np.mean( np.sqrt(np.sum( (Tensor_pos.cpu().detach().numpy().squeeze()-a_set["3d_pos"])**2, axis=1 )) )

                axs[0,6].set_axis_off()
                axs[0,6].set_title("Ground Truth {:.2f}".format(ori_error))  
                _2d_pos = a_set["2d_pos"]
                _2d_pos[:,0] = (_2d_pos[:,0] - ROI[2]) * a_set["scale_factors"][1]
                _2d_pos[:,1] = (_2d_pos[:,1] - ROI[0])* a_set["scale_factors"][0]
                plot_joint(img, _2d_pos, axs[0, 6])

                fig.savefig(os.path.join(new_dir, f[:8] + ".jpg"))
                # np.save(os.path.join(new_dir, f[:8] + "hm.npy"), hms)

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
    invalid_depth = 0
    depth_max = 1000
    scale_factors = np.array([0.,0.])
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
                depth_norm = a_set["depth_norm"]
                depth_with_img = np.dstack((depth_norm, img)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None, ...].to(device)
                
                center = center_from_heatmap(result.squeeze())

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

                # Regress the joint position
                depth_with_img = np.dstack((cropped_depth_norm, cropped_hand/255)).transpose((2,0,1))
                depth_with_img = depth_with_img.astype("float32")
                Img = torch.from_numpy(depth_with_img)
                Img = Img[None,...].to(device)
                heatmaps = JLo(Img)
                heatmaps_numpy = heatmaps.squeeze().cpu().detach().numpy()
                pred_pos = naive_pos_from_heatmap(heatmaps_numpy)

                pred_pos_plot = pred_pos.copy()
                pred_pos[:,0] = pred_pos[:,0]/scale_factors[1] + ROI[2]
                pred_pos[:,1] = pred_pos[:,1]/scale_factors[0] + ROI[0]

                pos0 = back_project(pred_pos[0], depth)
                pos5 = back_project(pred_pos[5], depth)
                pos9 = back_project(pred_pos[9], depth)
                pos5 -= pos0
                pos9 -= pos0
                z_body_frame = np.cross(pos5, pos9)

                # Normalize the y axis and z axis in the body frame
                y_body_frame = pos9 / np.linalg.norm(pos9)
                z_body_frame = z_body_frame / np.linalg.norm(z_body_frame)
                x_body_frame = np.cross(y_body_frame, z_body_frame).reshape(-1,1)
                
                y_body_frame = y_body_frame.reshape((-1,1))
                z_body_frame = z_body_frame.reshape((-1,1))
                
                R = np.array([[0,0,1],[1,0,0],[0,1,0]]) @ \
                    np.hstack((y_body_frame, z_body_frame, x_body_frame)).T
                R_inv = np.linalg.inv(R)

                # plot the original annotations
                _2d_pos = a_set["2d_pos"]
                _3d_pos = a_set["3d_pos"]

                # transform the pos array to the cropped image space
                _2d_pos_plot = _2d_pos.copy()
                _2d_pos_plot[:, 0] = (_2d_pos_plot[:, 0] - ROI[2])*scale_factors[1]
                _2d_pos_plot[:, 1] = (_2d_pos_plot[:, 1] - ROI[0])*scale_factors[0]

                axs[0, 3].set_axis_off()
                axs[0, 3].set_title("Keypoints GT")
                axs[0, 3].imshow(cropped_hand)
                for j in range(_2d_pos_plot.shape[0]):
                    axs[0, 3].scatter(_2d_pos_plot[j,0], _2d_pos_plot[j,1])
                
                # Plot the 2-D links results
                fingertip_indices = [4, 8, 12, 16, 20]
                error = np.mean(np.sqrt(np.sum((_2d_pos-pred_pos[fingertip_indices])**2, axis=1)))
                axs[0, 4].set_axis_off()
                axs[0, 4].set_title("2D Links {:.3f}".format(error))  
                plot_joint(cropped_hand, pred_pos_plot, axs[0, 4])
                
                # Plot the 3-D links results
                pre_output = PRe(Img)
                pred_3d_pos = decoder(pre_output["pos"])
                pred_3d_pos = pred_3d_pos.detach().cpu().view(-1, 3)
                pred_3d_pos_numpy = 1000*pred_3d_pos.numpy()
                pred_3d_pos_numpy = np.vstack((np.zeros(3), pred_3d_pos_numpy))
                pred_3d_pos_numpy = (R_inv @ pred_3d_pos_numpy.T).T
                pred_3d_pos_numpy += pos0

                error = np.mean(np.sqrt(np.sum((pred_3d_pos_numpy[fingertip_indices]-_3d_pos)**2, axis=-1)))
                accumulated_3d_error += error
                proj_pred_3d_pos = project2plane(pred_3d_pos_numpy)
                proj_pred_3d_pos_plot = proj_pred_3d_pos
                proj_pred_3d_pos_plot[:,0] = (proj_pred_3d_pos_plot[:,0] - ROI[2]) * scale_factors[1]
                proj_pred_3d_pos_plot[:,1] = (proj_pred_3d_pos_plot[:,1] - ROI[0]) * scale_factors[0]

                axs[0, 5].set_axis_off()
                axs[0, 5].set_title("3D Links {:.3f}".format(error))
                plot_joint(cropped_hand, proj_pred_3d_pos, axs[0, 5])

                # Plot the heatmaps of different parts
                counter = 0
                for r in range(1, plot_rows):
                    for c in range(plot_cols):
                        hm = cv2.resize(heatmaps_numpy[counter], tuple(config.cropped_size))
                        heatmap = Heatmap(hm)

                        composite = (alpha*cropped_hand + (1-alpha)*heatmap[...,::-1]).astype(np.uint8)
                        axs[r, c].set_axis_off()
                        axs[r, c].set_title(Part_Namelist[counter])
                        axs[r, c].imshow(composite)
                        
                        counter += 1
                        if counter == num_parts:
                            break
                
                fig.savefig(os.path.join(new_dir, f[:5] + ".jpg"))
                plt.close(fig)

                sampled_in_folder += 1

                if (sampled_in_folder > config.samples_per_folder):
                    break
        
            already_sampled += sampled_in_folder
            if (sampled_in_folder > config.test_samples):
                break
    
    averaged_error = accumulated_3d_error/sampled_in_folder
    np.savetxt(os.path.join(config.test_output_dir, "error_DAE1000.txt"), np.asarray([averaged_error]))


        
