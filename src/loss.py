# Python Libraries
import numpy as np

# Pytorch Libraris
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from src.loadConfig import loadConfig

class HALCriterion(object):
    """
    Get the loss of the output heatmap.
    """
    def __init__(self):
        """
        Constructor.
        """
        config = loadConfig()
        self.heatmap_size = config.heatmap_size
        self.weights = config.loss_weights[:2]

    def __call__(self, x, ground_truth, interm=None):
        """
        Get the Euclid distance^2 (L2 LOSS) between x and ground_truth.
        ---------------------------------------------------
        Args,
            x:              tensor (N x 1 x h x w), the predicted confidence map.
            ground_truth:   tensor (N x 1 x h x w), the groundtruth confidence map.
            interm:         list [k x 1], the intermidate result of the predicted confidence map.
        """
        loss = 0
        x = F.interpolate(x, size=self.heatmap_size, mode='bilinear')
        assert(x.shape == ground_truth.shape),  "Heatmap size mismatch!"
        
        loss += self.weights[0] * 1/x.shape[0] * torch.sum((x - ground_truth)**2)

        # for i, each in enumerate(interm):
        #     y = F.interpolate(each, size=self.heatmap_size, mode="bilinear")
        #     loss += self.weights[1] * 1/y.shape[0] * torch.sum((y - ground_truth)**2)

        return  loss


class JORCriterion(object):
    """
    Get the loss of the output heatmap and 3d joint position.
    """
    def __init__(self):
        """
        Constructor.
        """
        config = loadConfig()
        self.heatmap_size = [64, 64]
        self.weights = config.loss_weights[2:]

    def __call__(self, hm, interm_pos, pos, gt_heatmap, gt_pos):
        """
        Get the Euclid distance^2 (L2 LOSS) between x and ground_truth heatmap 
        & L2 loss between predicted position and ground_truth pos.
        ---------------------------------------------------
        Args,
            hm:             tensor (N x 1 x h x w), the predicted confidence map.
            interm_pos:     tensor (N x 63 [21x3]), the intermediate output of 3d joint positions.
            pos:            tensor (N x 63 [21x3] ), the predicted joint 3d positions.
            gt_heatmap:     tensor (21 x { N x 1 x H x W }), the groundtruth confidence map.
            gt_pos:         tensor (N X 63), the groundtruth joint 3d positions.
        """
        loss = 0
        hm = F.interpolate(hm, size=self.heatmap_size, mode='bilinear')

        assert(hm.shape == gt_heatmap.shape),  "Heatmap size mismatch!"
        
        # loss1 = self.weights[0] * 1/(21*hm.shape[0]) * torch.sum((hm - gt_heatmap)**2) 
        loss2 = self.weights[1] * 1/(21*pos.shape[0]) * torch.sum((pos - gt_pos)**2)

        # for each in interm_pos:
        #     loss += self.weights[2] * 1/pos.shape[0] * torch.sum((pos - gt_pos)**2)

        return loss2, loss2 