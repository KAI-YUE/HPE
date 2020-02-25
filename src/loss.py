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
        self.weights = config.loss_weights[1:]

    def __call__(self, hm, pos, gt_heatmap, gt_pos):
        """
        Get the Euclid distance^2 (L2 LOSS) between x and ground_truth heatmap 
        & L2 loss between predicted position and ground_truth pos.
        ---------------------------------------------------
        Args,
            hm:             tensor (N x 1 x h x w), the predicted confidence map.
            interm_pos:     tensor (N x 63 [21x3]), the intermediate output of 3d joint positions.
            gt_heatmap:     tensor (21 x { N x 1 x H x W }), the groundtruth confidence map.
        """
        # hm = F.interpolate(hm, size=self.heatmap_size, mode='bilinear')

        assert(hm.shape == gt_heatmap.shape),  "Heatmap size mismatch!"
        
        loss1 = self.weights[0] * 1/(hm.shape[0]) * torch.sum((hm - gt_heatmap)**2) 
        loss2 = self.weights[1] * 1/(hm.shape[0]) * torch.sum(torch.sqrt(torch.sum((pos-gt_pos)**2, dim=3)))
        # loss2 = 0

        return loss1+loss2, loss2