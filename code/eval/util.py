import math

import numpy as np
import torch
from six.moves import urllib
import os

from torch import nn
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim


def download_and_extract(filepath, save_dir):
    filename = filepath.split('/')[-1]
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        print('Already downloaded')
    else:
        urllib.request.urlretrieve(filepath, save_path)
        print(f'Successfully downloaded to {save_path}')


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.to_ts = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def forward(self, output, target, device):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        x_hat = output["x_hat"][0]
        x_hat = self.to_pil(x_hat)
        x_hat = x_hat.convert('L')
        x_hat = self.to_ts(x_hat)
        x_hat = x_hat.to(device)
        target = target[0]
        mse = self.mse(x_hat, target)
        out["mse_loss"] = mse

        x_hat = np.array(torch.squeeze(x_hat, 0).cpu())
        target = np.array(torch.squeeze(target, 0).cpu())
        ssim_loss = ssim(x_hat, target, multichannel=False)
        out["ssim_loss"] = ssim_loss

        x_hat = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x_hat), 0), 0)
        target = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(target), 0), 0)
        msssim_loss = ms_ssim(x_hat, target, data_range=1)
        out["msssim_loss"] = msssim_loss

        return out


if __name__ == '__main__':
    from eval.params import model_urls
    save_dir = '/home/kxfeng/checkpoint/'
    for model in model_urls:
        model_dir = os.path.join(save_dir, model)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for i in model_urls[model]['mse']:
            download_and_extract(model_urls[model]['mse'][i], model_dir)