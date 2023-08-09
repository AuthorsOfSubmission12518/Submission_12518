import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class img_folder(Dataset):
    def __init__(self, root_path, gt_path):
        super(img_folder, self).__init__()

        self.root_path = root_path
        self.gt_path = gt_path

        self.img_list = []

        for seq in os.listdir(self.root_path):
            for sub_seq in os.listdir(os.path.join(self.root_path, seq)):
                self.img_list.append({
                    'recon': os.path.join(self.root_path, seq, sub_seq, f'im1.png'),
                    'gt': os.path.join(self.gt_path, seq, sub_seq, f'im3.png')
                })
                self.img_list.append({
                    'recon': os.path.join(self.root_path, seq, sub_seq, f'im11.png'),
                    'gt': os.path.join(self.gt_path, seq, sub_seq, f'im4.png')
                })
                self.img_list.append({
                    'recon': os.path.join(self.root_path, seq, sub_seq, f'im21.png'),
                    'gt': os.path.join(self.gt_path, seq, sub_seq, f'im5.png')
                })

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        gt_path = img_path['gt']
        img_path = img_path['recon']

        img = Image.open(img_path)
        img_ts = self.to_tensor(img)
        img_ts = torch.squeeze(img_ts, 0)
        img_ts = torch.stack([img_ts, img_ts, img_ts])

        gt = Image.open(gt_path).convert('L')
        gt_ts = self.to_tensor(gt)
        return img_ts, gt_ts

    def __len__(self):
        return len(self.img_list) // 10
        # return 100
