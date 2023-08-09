import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop


# class spk_provider(Dataset):
#     def __init__(self, spk_dir, gt_dir, mode='train'):
#         super(spk_provider, self).__init__()
#
#         self.spk_path = spk_dir
#         self.gt_path = gt_dir
#         self.mode = mode
#
#         self.spk_gt_list = []
#
#         for seq in os.listdir(spk_dir):
#             for sub_seq in os.listdir(os.path.join(spk_dir, seq)):
#                 if (mode == 'train' and int(sub_seq.split('.')[0]) % 10 != 0) or (mode == 'test' and int(sub_seq.split('.')[0]) % 10 == 0):
#                     self.spk_gt_list.append({
#                         'spk_path': os.path.join(spk_dir, seq, sub_seq),
#                         'gt_path': os.path.join(gt_dir, seq, sub_seq.split('.')[0])
#                     })
#
#         self.to_tensor = ToTensor()
#         self.crop = CenterCrop((256, 256))
#
#     def __getitem__(self, idx):
#         spk_gt = self.spk_gt_list[idx]
#         spk_path = spk_gt['spk_path']
#         gt_path = spk_gt['gt_path']
#
#         sub_idx = random.randint(3, 5)
#
#         spk_voxel = np.load(spk_path)
#         spk_voxel = spk_voxel[10 * (sub_idx - 3):10 * (sub_idx - 3) + 41]
#         spk_voxel = torch.from_numpy(spk_voxel).float()
#
#         gt = Image.open(os.path.join(gt_path, f'im{sub_idx}.png')).convert('L')
#         gt = self.to_tensor(gt)
#
#         return self.crop(spk_voxel), self.crop(gt)
#
#     def __len__(self):
#         return len(self.spk_gt_list) // 100


class spk_provider(Dataset):
    def __init__(self, spk_dir, gt_dir, mode='train'):
        super(spk_provider, self).__init__()

        self.spk_path = spk_dir
        self.gt_path = gt_dir
        self.mode = mode

        self.spk_gt_list = []

        for seq_subseq in [
            '00001-0677',
            '00033-0231',
            '00033-0532',
            '00045-0133',
            '00050-0511',
            '00050-0580',
            '00052-0197',
            '00057-0359',
            '00072-0779',
            '00080-0119'
        ]:
            seq = seq_subseq.split('-')[0]
            sub_seq = seq_subseq.split('-')[1]
            for i in range(0, 21):
                self.spk_gt_list.append({
                    'spk_path': os.path.join(spk_dir, seq, f'{sub_seq}.npy'),
                    'gt_path': os.path.join(gt_dir, seq, sub_seq.split('.')[0]),
                    'im_idx': i
                })

        # for seq in os.listdir(spk_dir):
        #     for sub_seq in os.listdir(os.path.join(spk_dir, seq)):
        #         if (mode == 'train' and int(sub_seq.split('.')[0]) % 10 != 0) or (mode == 'test' and int(sub_seq.split('.')[0]) % 10 == 0):
        #             self.spk_gt_list.append({
        #                 'spk_path': os.path.join(spk_dir, seq, sub_seq),
        #                 'gt_path': os.path.join(gt_dir, seq, sub_seq.split('.')[0])
        #             })
        self.crop = CenterCrop((256, 256))
        self.to_tensor = ToTensor()

    def __getitem__(self, idx):
        spk_gt = self.spk_gt_list[idx]
        spk_path = spk_gt['spk_path']
        gt_path = spk_gt['gt_path']
        im_idx = spk_gt['im_idx']
        seq = '00036'
        sub_seq = '0606'
        sub_idx = 4

        spk_path = f'/data/kxfeng/vimeo_septuplet_spike/{seq}/{sub_seq}.npy'
        gt_path = f'/data/klin/vimeo_septuplet/sequences/{seq}/{sub_seq}'

        # sub_idx = random.randint(3, 5)


        spk_voxel = np.load(spk_path)
        # spk_voxel = spk_voxel[im_idx: im_idx + 41]
        # spk_voxel = torch.from_numpy(spk_voxel).float()
        spk_voxel = spk_voxel[10 * (sub_idx - 3):10 * (sub_idx - 3) + 41]
        spk_voxel = torch.from_numpy(spk_voxel).float()

        gt = Image.open(os.path.join(gt_path, f'im{sub_idx}.png')).convert('L')
        gt = self.to_tensor(gt)

        return self.crop(spk_voxel), self.crop(gt), os.path.join(gt_path, f'im{sub_idx}.png')
        # return self.crop(spk_voxel), spk_path, im_idx

    def __len__(self):
        return 1


class spk_provider_real(Dataset):
    def __init__(self, spk_dir):
        super().__init__()
        self.spk_path = spk_dir

        self.spk_list = []

        for item in os.listdir(spk_dir):
            if item.split('-')[0] != 'ballon':
                continue
            self.spk_list.append(os.path.join(spk_dir, item))
        self.crop = CenterCrop((256, 256))
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        spk_path = self.spk_list[idx]
        file_name = spk_path.split('/')[-1]
        spk_voxel = np.load(spk_path)
        spk_voxel = torch.from_numpy(spk_voxel).float()
        return file_name, self.crop(spk_voxel)

class spk_provider_fake(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        spk = torch.zeros((41, 256, 256))
        gt = torch.zeros(1, 256, 256)
        return spk, gt

    def __len__(self):
        return 6


class spk_provider_small(spk_provider):
    def __getitem__(self, idx):
        spk_gt = self.spk_gt_list[idx]
        spk_path = spk_gt['spk_path']
        gt_path = spk_gt['gt_path']

        sub_idx = random.randint(3, 5)

        spk_voxel = np.load(spk_path)
        spk_voxel = spk_voxel[10 * (sub_idx - 1) - 12:10 * (sub_idx - 1) + 15]
        spk_voxel = torch.from_numpy(spk_voxel).float()

        gt = Image.open(os.path.join(gt_path, f'im{sub_idx}.png')).convert('L')
        gt = self.to_tensor(gt)

        return spk_voxel, gt


class spk_provider_shortterm(spk_provider):
    def __getitem__(self, idx):
        spk_gt = self.spk_gt_list[idx]
        spk_path = spk_gt['spk_path']
        gt_path = spk_gt['gt_path']

        sub_idx = random.randint(2, 6)

        spk_voxel = np.load(spk_path)
        spk_voxel = spk_voxel[10 * (sub_idx - 1) - 5:10 * (sub_idx - 1) + 8]
        spk_voxel = torch.from_numpy(spk_voxel).float()

        gt = Image.open(os.path.join(gt_path, f'im{sub_idx}.png')).convert('L')
        gt = self.to_tensor(gt)

        return spk_voxel, gt
