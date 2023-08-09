import math

import torch
from torch import nn


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(torch.clamp(output["x_hat"], 0, 1), target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class RDDLoss(RateDistortionLoss):
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss1"] = self.mse(torch.clamp(output["x_hat1"], 0, 1), target)
        out["mse_loss"] = self.mse(torch.clamp(output["x_hat"], 0, 1), target)
        out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss1"] + out["mse_loss"]) + out["bpp_loss"]

        return out
