# Python Libraries
import numpy as np

# Pytorch Libraris
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from src.loadConfig import loadConfig

class HLoCriterion(object):
    """
    Get the loss of the output heatmap.
    """
    def __init__(self):
        """
        Constructor.
        """
        config = loadConfig()
        self.heatmap_size = [config.input_size[1], config.input_size[0]]
        self.weights = config.loss_weights[0]
        self.epsilon = torch.tensor(1e-8)

    def __call__(self, x, ground_truth):
        """
        Get the Euclid distance^2 (L2 LOSS) between x and ground_truth.
        ---------------------------------------------------
        Args,
            x:              tensor (N x 1 x h x w), the predicted confidence map.
            ground_truth:   tensor (N x 1 x h x w), the groundtruth confidence map.
        """
        loss = 0
        # x = F.interpolate(x, size=self.heatmap_size, mode='bilinear')
        assert(x.shape == ground_truth.shape),  "Heatmap size mismatch!"
        
        loss += self.weights * 1/x.shape[0] * torch.sum((x - ground_truth)**2)
        # loss = self.weights * 1/x.shape[0] * torch.sum(-(ground_truth*torch.log(x + self.epsilon) + \
        #                                             (1-ground_truth)*torch.log((1-x)+self.epsilon) ))
        
        return  loss


class PReCriterion(object):
    """
    Get the loss of the output heatmap and 3d joint position.
    """
    def __init__(self):
        """
        Constructor.
        """
        config = loadConfig()
        weights_level = [10., 8., 5., 3., 1.]
        self.weights = torch.tensor([[weights_level[0], weights_level[0], weights_level[1], weights_level[1],                                       # thumb
                                      weights_level[2], weights_level[2], weights_level[3], weights_level[3],                                       # thumb
                                      weights_level[0], weights_level[1], weights_level[1], weights_level[2], weights_level[3],                     # index
                                      weights_level[1], weights_level[1], weights_level[2], weights_level[3],                                       # middle
                                      weights_level[0], weights_level[0], weights_level[1], weights_level[1], weights_level[2], weights_level[3],   # ring
                                      weights_level[0], weights_level[0], weights_level[1], weights_level[1], weights_level[2], weights_level[3]    # little
                                      ]], device="cuda")   


    def __call__(self, networks_output, scale, theta_alpha):
        """
        Get the Euclid distance^2 (L2 LOSS) between x and ground_truth heatmap 
        & L2 loss between predicted position and ground_truth pos.
        ---------------------------------------------------
        Args,
            hm:             tensor (N x 1 x h x w), the predicted confidence map.
            interm_pos:     tensor (N x 63 [21x3]), the intermediate output of 3d joint positions.
            gt_heatmap:     tensor (21 x { N x 1 x H x W }), the groundtruth confidence map.
        """
        loss = 0.
        # loss += 1/scale.shape[0] * torch.sum((networks_output["scale"] - scale)**2)
        # loss += 1/theta_alpha.shape[0] * torch.sum((networks_output["theta"] - theta_alpha)**2) 
        loss += 1/theta_alpha.shape[0] * torch.sum(self.weights @ ((networks_output["theta"] - theta_alpha)**2).transpose(0,1))
       
        return loss