import math
import os.path
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

from recon_net.spk2imgnet import new_Spike_Net_v3 as recon_net


class Spk_Seq_Dataset(Dataset):
    def __init__(self, spk_dir, mode=0):
        super(Spk_Seq_Dataset, self).__init__()

        self.spk_dir = spk_dir
        self.spk_path_list = []

        for seq in os.listdir(spk_dir):
            if int(seq) % 3 == mode:
                for sub_seq in os.listdir(os.path.join(spk_dir, seq)):
                    path = os.path.join(spk_dir, seq, sub_seq)
                    self.spk_path_list.append(path)

    def __getitem__(self, idx):
        spk_path = self.spk_path_list[idx]
        seq = spk_path.split('/')[-2]
        sub_seq = spk_path.split('/')[-1].split('.')[0]

        spk_voxel = np.load(spk_path)
        spk_ts = torch.from_numpy(spk_voxel).float()

        return {
            'seq': seq,
            'sub_seq': sub_seq,
            'spk': spk_ts
        }

    def __len__(self):
        return len(self.spk_path_list)


if __name__ == '__main__':
    spk_dir = '/data2/kxfeng/vimeo_septuplet_spike/'
    save_dir = '/data2/kxfeng/vimeo_septuplet_recon/'
    load_checkpoint = '/home/kxfeng/iccv/checkpoint/recon_net.pth'
    device = sys.argv[1]
    mode = int(sys.argv[2])
    batch = 7
    num_workers = 4

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    net = net.to(device)
    net.load_state_dict(torch.load(load_checkpoint, map_location=device)['state_dict'])
    net.eval()

    dataset = Spk_Seq_Dataset(spk_dir, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device != "cpu")
    )
    to_img = ToPILImage()

    with torch.no_grad():
        for i, d in enumerate(dataloader):
            seq_list = d['seq']
            sub_seq_list = d['sub_seq']
            spk_voxel = d['spk'].to(device)

            for frame_idx in range(0, 21):
                spk = spk_voxel[:, frame_idx: frame_idx + 41, :, :]
                recon = net(spk)[0]

                for b in range(recon.shape[0]):
                    recon_b = recon[b]
                    recon_b = to_img(recon_b)

                    if not os.path.exists(os.path.join(save_dir, seq_list[b])):
                        os.mkdir(os.path.join(save_dir, seq_list[b]))
                    if not os.path.exists(os.path.join(save_dir, seq_list[b], sub_seq_list[b])):
                        os.mkdir(os.path.join(save_dir, seq_list[b], sub_seq_list[b]))

                    save_path = os.path.join(save_dir, seq_list[b], sub_seq_list[b], f'im{frame_idx + 1}.png')
                    print(save_path)
                    recon_b.save(save_path)

            # for b in range(recon.shape[0]):
            #     seq = seq_list[b]
            #     sub_seq = sub_seq_list[b]
            #     print(f'Finish generating {seq}/{sub_seq}')

