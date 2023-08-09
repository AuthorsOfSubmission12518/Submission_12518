import os
import json
import shutil
import sys
import time
import argparse

import math
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whole_net.joint_net import Joint_Net_with_lstm_att, Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, \
    Joint_Net_LIC_with_bilstm_att
from com_net.util.loss import RateDistortionLoss, RDDLoss

from pms.joint.recon_cheng128 import pms
from folder.spk_folder import spk_provider


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.__init__()


def cal_loss(a, b):

    a = 1 / torch.log10(a)
    b = 1 / torch.log10(b)

    return torch.nn.MSELoss()(a, b)


def test_one_epoch(
        model,
        test_dataloader):
    model.eval()

    ave = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            isi = d[0].to(pms['device'])
            img = d[1].to(pms['device'])
            # t = time.time()
            out_net = model(isi)
            # print(out_net['att_map'].shape)
            # ave.update(time.time() - t)
            loss = RateDistortionLoss()(out_net, img)
            print(loss['bpp_loss'].item())
            print(loss['mse_loss'].item())
            print(-10 * math.log10(loss['mse_loss'].item()))
            # ave.update(loss)
            from torchvision.transforms import ToPILImage
            img = ToPILImage()(torch.squeeze(out_net['x_hat']))
            img.save(f'/data/kxfeng/lic.png')
            # img.save(f'/data/kxfeng/lstm.png')
            # img.save(f'/data/kxfeng/bilstm.png')
    # print(ave.avg)
    # print('Ave: ', ave.avg, -10 * math.log10(ave.avg))


def main():
    test_dataset = spk_provider(pms['spk_path'], pms['gt_path'], 'test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pms['test_batch'],
        num_workers=pms['worker_num'],
        shuffle=False,
        pin_memory=(pms['device'] != "cpu")
    )

    # net = Joint_Net_LIC_with_bilstm_att(pms['load_checkpoint'], pms['device'])
    # net = Joint_Net_LIC_with_lstm_att(pms['load_checkpoint'], pms['device'])
    net = Joint_Net_LIC(pms['load_checkpoint'], pms['device'], None)
    net = net.to(pms['device'])

    checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
    print("Loading from ", pms['load_checkpoint'])
    net.load_state_dict(checkpoint["state_dict"])
    print("Loading finish")

    test_one_epoch(
        net,
        test_dataloader)


if __name__ == '__main__':

    print(f'current pid: {os.getpid()}')
    for l in [0.05]:
        pms['lmbda'] = l
        print(l)
        if not os.path.exists(f'/data/kxfeng/recon_seq_bilstm/{pms["lmbda"]}'):
            os.mkdir(f'/data/kxfeng/recon_seq_bilstm/{pms["lmbda"]}')
        pms['device'] = 'cuda:0'
        pms['batch'] = 2
        pms['test_batch'] = 1
        pms['worker_num'] = 1
        pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/lic/joint-{pms["lmbda"]}.pth'
        # pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/lic/lstm_att/joint-{pms["lmbda"]}.pth'
        # pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/lic/bilstm_att/joint-{pms["lmbda"]}.pth'
        # print(json.dumps(pms, indent=2))
        main()
