import torch
import torch.nn as nn

import numpy as np

import sys
sys.path.append('../../../code')
sys.path.append('../../models')

from utils.sketch_utils import Sketches



class SilhouetteLoss(nn.Module):
    def __init__(self, silhouette_canvases):
        super().__init__()
        self.silhouette_canvases = silhouette_canvases

    def forward(self, weights_sum):
        #maximze log probability of weights_sum
        #weights_sum: (B, H, W)
        #self.sihouette_canvases: (B, H, W)

        #clamp weights_sum to avoid log(0)
        weights_sum = weights_sum.clamp(1e-5, 1 - 1e-5)
        loss = -torch.mean(torch.log(weights_sum) * self.silhouette_canvases)

        return loss

