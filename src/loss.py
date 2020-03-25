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
        weights_level = [1., 1., 1., 1., 1.]
        self.weights = [1, 20, 100]

    def __call__(self, networks_output, data):
        """
        Get the Euclid distance^2 (L2 LOSS) between predicted position and ground_truth pos.
        ---------------------------------------------------
        """

        loss = 0.
        # loss += 1/scale.shape[0] * torch.sum((networks_output["scale"] - scale)**2)
        # loss += 1/theta_alpha.shape[0] * torch.sum((networks_output["theta"] - theta_alpha)**2) 
        pos_loss = self.weights[0] * torch.mean(torch.sqrt(torch.sum((networks_output["pos"] - data["pos"])**2, dim=-1)))
        # loss += self.weights[1] * 1/data["theta_alpha"].shape[0] * torch.sum((networks_output["theta"] - data["theta_alpha"])**2)
        # loss += self.weights[2] * 1/data["R_inv"].shape[0] * torch.sum((networks_output["R_inv"] - data["R_inv"])**2)
        loss += pos_loss

        return dict(loss=loss, pos_loss=pos_loss)