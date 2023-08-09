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
        self.extractor = Feature_Extractor(in_channels=in_channels, features=features, out_channels=features, channel_step=1, num_of_layers=12)

        self.mask0 = Fusion_mask_v1(features=features)
        self.mask1 = Fusion_mask_v1(features=features)
        self.mask3 = Fusion_mask_v1(features=features)
        self.mask4 = Fusion_mask_v1(features=features)

        self.rec_conv0 =nn.Conv2d(in_channels=5*features, out_channels=3*features, kernel_size=3, padding=1)
        self.rec_conv1 =nn.Conv2d(in_channels=3*features, out_channels=features, kernel_size=3, padding=1)
        self.rec_conv2 =nn.Conv2d(in_channels=features, out_channels=1, kernel_size=3, padding=1)
        self.rec_relu = nn.ReLU()

        self.pcd_align = Easy_PCD(nf=features, groups=8)
        self.win_r = win_r
        self.win_step = win_step

    def forward(self, x):

        block0 = x[:,0:2*self.win_r+1,:,:]
        block1 = x[:,self.win_step:self.win_step+2*self.win_r+1,:,:]
        block2 = x[:,2*self.win_step:2*self.win_step+2*self.win_r+1,:,:]
        block3 = x[:,3*self.win_step:3*self.win_step+2*self.win_r+1,:,:]
        block4 = x[:,4*self.win_step:4*self.win_step+2*self.win_r+1,:,:]

        block0_out, est0 = self.extractor(block0)
        block1_out, est1 = self.extractor(block1)
        block2_out, est2 = self.extractor(block2)
        block3_out, est3 = self.extractor(block3)
        block4_out, est4 = self.extractor(block4)

        aligned_block0_out = self.pcd_align(block0_out, block2_out)
        aligned_block1_out = self.pcd_align(block1_out, block2_out)
        aligned_block3_out = self.pcd_align(block3_out, block2_out)
        aligned_block4_out = self.pcd_align(block4_out, block2_out)

        mask0 = self.mask0(aligned_block0_out, block2_out)
        mask1 = self.mask1(aligned_block1_out, block2_out)
        mask3 = self.mask3(aligned_block3_out, block2_out)
        mask4 = self.mask4(aligned_block4_out, block2_out)

        out = torch.cat((aligned_block0_out*mask0, aligned_block1_out*mask1), 1)
        out = torch.cat((out, block2_out), 1)
        out = torch.cat((out, aligned_block3_out*mask3), 1)
        out = torch.cat((out, aligned_block4_out*mask4), 1)

        out = self.rec_relu(self.rec_conv0(out))
        out = self.rec_relu(self.rec_conv1(out))
        out = self.rec_conv2(out)

        return out, est0, est1, est2, est3, est4


# if __name__ == '__main__':
    # device = 'cuda:5'
    # x = torch.from_numpy(np.load('/home/kxfeng/iccv/spk_3.npy')).float()
    # x = torch.unsqueeze(x, 0)
    # x = x.to(device)
    # net = new_Spike_Net_v3(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    # net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/home/kxfeng/tmp/model_061.pth', map_location=device).items()}, )
    #
    # parameters = {
    #     n
    #     for n, p in net.named_parameters()
    #     if p.requires_grad
    # }
    # params_dict = dict(net.named_parameters())
    # optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4)
    #
    # net = net.to(device)
    # mse = nn.MSELoss()
    # z = torchvision.transforms.ToTensor()(Image.open('/home/kxfeng/iccv/51.png'))
    # optimizer.step()
    #
    # print('..')


    # net.eval()
    # y = net(x)
    # y = torch.squeeze(y[0], 0)
    #
    # t = torchvision.transforms.ToPILImage()
    # y = t(y)

    # y.save('/home/kxfeng/iccv/4.png')

    # mse = nn.MSELoss()
    # y = Image.open('/home/kxfeng/iccv/3.png')
    # z = Image.open('/home/kxfeng/iccv/31.png')
    #
    # t = torchvision.transforms.ToTensor()
    # y = t(y)
    # z = t(z)
    # b = float('inf')
    # b_i = 0
    # for i in range(200, 500, 1):
    #     p = y ** (1 / (i / 100))
    #     print(i / 100, mse(p, z).item())
    #     if mse(p, z).item() < b:
    #         b = mse(p, z).item()
    #         b_i = i / 100
    #
    # print(b_i, b)
