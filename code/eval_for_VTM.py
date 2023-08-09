import math
import os.path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from recon_net.spk2imgnet import new_Spike_Net_v3 as recon_net


class VTM_Dataset(Dataset):
    def __init__(self, spk_dir, gt_dir):
        super(VTM_Dataset, self).__init__()

        self.sample_file_path = '/data2/kxfeng/vvc+spk2imgnet/sample.txt'
        self.sample_list = {}
        with open(self.sample_file_path) as f:
            lines = f.readlines()
            for l in lines:
                seq = l.split('\t')[0]
                sub_seq = l.split('\t')[1]
                im_idx = l.split('\t')[2].split('\n')[0]

                if seq not in self.sample_list:
                    self.sample_list[seq] = {}
                self.sample_list[seq][sub_seq] = im_idx

        self.img_path_list = []
        for seq in os.listdir(spk_dir):
            for sub_seq in os.listdir(os.path.join(spk_dir, seq)):
                im_idx = self.sample_list[seq][sub_seq]
                spk_path = os.path.join(spk_dir, seq, sub_seq)
                gt_path = os.path.join(gt_dir, seq, sub_seq.split(".")[0], f'im{im_idx}.png')

                self.img_path_list.append({
                    'spk_path': spk_path,
                    'gt_path': gt_path
                })

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        spk_path = img_path['spk_path']
        gt_path = img_path['gt_path']

        spk_voxel = np.load(spk_path)
        spk_voxel_ts = torch.from_numpy(spk_voxel).float()
        spk_voxel_ts /= 255

        gt_img = Image.open(gt_path).convert('L')
        gt_img_ts = self.transform(gt_img)

        return spk_voxel_ts, gt_img_ts

    def __len__(self):
        return len(self.img_path_list)


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


if __name__ == '__main__':
    spk_dir = '/data2/kxfeng/vvc+spk2imgnet/vvc_recon_spk'
    gt_dir = '/backup1/klin/vimeo_septuplet/sequences/'
    load_checkpoint = '/home/kxfeng/iccv/checkpoint/recon_net.pth'
    device = 'cuda:7'
    batch = 6
    num_workers = 3

    qp_list = [22, 27, 32, 37, 42, 47, 52, 57]

    net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    net = net.to(device)
    net.load_state_dict(torch.load(load_checkpoint, map_location=device)['state_dict'])
    net.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        for qp in qp_list:
            spk_qp_dir = os.path.join(spk_dir, str(qp))
            dataset = VTM_Dataset(spk_qp_dir, gt_dir)

            dataloader = DataLoader(
                dataset,
                batch_size=batch,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=(device != "cpu")
            )
            ave = AverageMeter()

            for i, (spk, gt) in enumerate(dataloader):
                spk = spk.to(device)
                gt = gt.to(device)

                recon = net(spk)[0]
                out_criterion = criterion(recon, gt)
                ave.update(out_criterion.item())

            print(f'{qp}\tMSE:{round(ave.avg, 4)}\tPSNR:{round(-10 * math.log10(ave.avg), 4)}')




