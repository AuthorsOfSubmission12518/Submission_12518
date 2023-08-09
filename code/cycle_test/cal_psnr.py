import math
import os.path

import numpy as np
from PIL import Image

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
    root_dir = 'G:\\vimeo_septuplet\\vimeo_septuplet_spike_91'

    model_list = ['lic', 'lstm', 'bilstm']
    lmbda_list = ['0.05', '0.025', '0.013', '0.0067']

    base_ave = AverageMeter()
    model_ave_list = {}
    for m in model_list:
        model_ave_list[m] = {}
        for l in lmbda_list:
            model_ave_list[m][l] = AverageMeter()

    for seq in ['00001']:
        for sub_seq in [
            '0001',
            '0002',
            '0003',
            '0004',
            '0005',
            '0006',
            '0007',
            '0008',
            '0009',
            '0010',
        ]:
            gt_path = os.path.join(root_dir, seq, sub_seq, 'frames', '51.png')
            gt = Image.open(gt_path)
            gt = np.array(gt)[:, 96: -96]
            gt = gt / 255

            base_path = os.path.join(root_dir, seq, sub_seq, 'base.png')
            base = Image.open(base_path)
            base = np.array(base)
            base = base / 255

            base_mse = np.average((gt - base) ** 2)
            base_ave.update(base_mse)

            for item in os.listdir(os.path.join(root_dir, seq, sub_seq, 're_recon_scene')):
                m = item.split('-')[0]
                l = item.split('-')[1].split('.png')[0]

                path = os.path.join(root_dir, seq, sub_seq, 're_recon_scene', item)
                img = Image.open(path)
                img = np.array(img)
                img = img / 255

                img_mse = np.average((base - img) ** 2)
                model_ave_list[m][l].update(img_mse)

    print('base', base_ave.avg, -10 * math.log10(base_ave.avg))
    for m in model_list:
        print()
        for l in lmbda_list:
            try:
                print(m, l, model_ave_list[m][l].avg, -10 * math.log10(model_ave_list[m][l].avg))
            except KeyError:
                print(m, l)

