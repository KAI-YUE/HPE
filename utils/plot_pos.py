# Python Libraries
import matplotlib.pyplot as plt

# My Libraries
# import 

def plot_joint(img, pos_arr, axs=None):
    """
    Plot the landmarks on the hand.
    ---------------------------------------------------
    Args,
        img:      ndarray (H x W x 3): the image of the color image on depth. [0, 255]
        pos_arr:  ndarray (21 x 2): the 2d position array.
        axs:      the axes of the figure. If set to None, a new axs will be created.
    """

    if axs is None:
        fig = plt.figure()
        axs = fig.gca()
    
    axs.imshow(img)
    

    # Plot the wrist center 
    axs.scatter(pos_arr[0,0], pos_arr[0,1], color='k')

    # Plot the little keypoints
    red = "#FF3333"
    for i in range(17, 21):
        axs.scatter(pos_arr[i,0], pos_arr[i,1], color=red)
        j = i-1 if i>17 else 0
        axs.plot([pos_arr[j,0], pos_arr[i,0]], [pos_arr[j,1], pos_arr[i, 1]], color=red, linewidth=3.5)
    
    # Plot the ring keypoints
    violet = "#FF66FF"
    for i in range(13, 17):
        axs.scatter(pos_arr[i,0], pos_arr[i,1], color=violet)
        j = i-1 if i>13 else 0
        axs.plot([pos_arr[j,0], pos_arr[i,0]], [pos_arr[j,1], pos_arr[i, 1]], color=violet, linewidth=3.5)

    # Plot the middle keypoints
    blue = "#66B2FF"
    for i in range(9, 13):
        axs.scatter(pos_arr[i,0], pos_arr[i,1], color=blue)
        j = i-1 if i>9 else 0
        axs.plot([pos_arr[j,0], pos_arr[i,0]], [pos_arr[j,1], pos_arr[i, 1]], color=blue, linewidth=3.5)

    # Plot the index keypoints
    cyan = "#66FFFF"
    for i in range(5, 9):
        axs.scatter(pos_arr[i,0], pos_arr[i,1], color=cyan)
        j = i-1 if i>5 else 0
        axs.plot([pos_arr[j,0], pos_arr[i,0]], [pos_arr[j,1], pos_arr[i, 1]], color=cyan, linewidth=3.5)
    
    
    # Plot the thumb keypoints
    green = "#66FF66"
    for i in range(1, 5):
        axs.scatter(pos_arr[i,0], pos_arr[i,1], color=green)
        axs.plot([pos_arr[i-1,0], pos_arr[i,0]], [pos_arr[i-1,1], pos_arr[i, 1]], color=green, linewidth=3.5)