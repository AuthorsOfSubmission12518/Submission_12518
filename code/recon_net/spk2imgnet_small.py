import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image
from torch import optim

from recon_net.extractor import Feature_Extractor
from recon_net.pcd import Easy_PCD


class Fusion_mask_v1(nn.Module):
    def __init__(self, features):
        super(Fusion_mask_v1, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2*features, out_channels = features, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=features, out_channels = features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels = features, kernel_size=3, padding=1)
        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.sig = nn.Sigmoid()

    def forward(self, ref, key):
        fea = torch.cat((ref, key), 1)
        fea = self.conv2(self.prelu1(self.conv1(self.prelu0(self.conv0(fea)))))
        mask = self.sig(fea)
        return mask


class new_Spike_Net_v3(nn.Module):
    def __init__(self, in_channels, features, out_channels, win_r, win_step):
        super(new_Spike_Net_v3, self).__init__()
        self.extractor = Feature_Extractor(in_channels=in_channels, features=features, out_channels=features, channel_step=1, num_of_layers=6)

        self.mask0 = Fusion_mask_v1(features=features)
        self.mask2 = Fusion_mask_v1(features=features)

        self.rec_conv0 =nn.Conv2d(in_channels=3*features, out_channels=features, kernel_size=3, padding=1)
        self.rec_conv1 =nn.Conv2d(in_channels=features, out_channels=1, kernel_size=3, padding=1)
        self.rec_relu = nn.ReLU()

        self.pcd_align = Easy_PCD(nf=features, groups=8)
        self.win_r = win_r
        self.win_step = win_step

    def forward(self, x):

        block0 = x[:,0:2*self.win_r+1,:,:]
        block1 = x[:,self.win_step:self.win_step+2*self.win_r+1,:,:]
        block2 = x[:,2*self.win_step:2*self.win_step+2*self.win_r+1,:,:]

        block0_out, est0 = self.extractor(block0)
        block1_out, est1 = self.extractor(block1)
        block2_out, est2 = self.extractor(block2)

        aligned_block0_out = self.pcd_align(block0_out, block1_out)
        aligned_block2_out = self.pcd_align(block2_out, block1_out)

        mask0 = self.mask0(aligned_block0_out, block1_out)
        mask2 = self.mask2(aligned_block2_out, block1_out)

        out = torch.cat((aligned_block0_out*mask0, block1_out), 1)
        out = torch.cat((out, aligned_block2_out*mask2), 1)

        out = self.rec_relu(self.rec_conv0(out))
        out = self.rec_conv1(out)

        return out, est0, est1, est2
