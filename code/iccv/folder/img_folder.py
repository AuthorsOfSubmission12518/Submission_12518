import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class img_folder(Dataset):
    def __init__(self, root_path, mode='train'):
        super(img_folder, self).__init__()

        self.root_path = root_path
        self.mode = mode

        self.img_list = []

        for seq in os.listdir(self.root_path):
            for sub_seq in os.listdir(os.path.join(self.root_path, seq)):
                if (self.mode == 'train' and int(sub_seq) % 10 != 0) or (self.mode == 'test' and int(sub_seq) % 10 == 0):
                    for i in range(1, 8):
                        self.img_list.append(os.path.join(self.root_path, seq, sub_seq, f'im{i}.png'))

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('L')
        img_ts = self.to_tensor(img)
        return img_ts

    def __len__(self):
        return len(self.img_list)
