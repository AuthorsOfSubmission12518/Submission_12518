import os.path
import random
import time

import torch
import numpy as np

from PIL import Image

from spk2imgnet import new_Spike_Net_v3 as spk2imgnet


def mse(a, b):
    return np.average((a - b) ** 2)


if __name__ == '__main__':
    # checkpoint_path = '/home/kxfeng/iccv/checkpoint/recon_net.pth'
    # device = 'cuda:6'
    # gt_dir = '/backup5/kxfeng/vimeo_septuplet/sequences'
    # spike_dir = '/backup5/kxfeng/vimeo_septuplet_spike'
    # recon_dir = '/backup5/kxfeng/vimeo_septuplet_spike_recon'
    # log_path = '/backup5/kxfeng/vimeo_septuplet_spike_recon/log.txt'
    checkpoint_path = 'K:\\vimeo_septuplet_recon_via_spk2imgnet\\checkpoint.pth'
    device = 'cuda:0'
    gt_dir = 'G:\\vimeo_septuplet\\vimeo_septuplet\\sequences'
    spike_dir = 'K:\\vimeo_septuplet_spike'
    recon_dir = 'K:\\vimeo_septuplet_recon_via_spk2imgnet'
    log_path = 'K:\\vimeo_septuplet_recon_via_spk2imgnet\\log.txt'
    if not os.path.exists(recon_dir):
        os.mkdir(recon_dir)

    model = spk2imgnet(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    model.eval()

    with open(log_path, 'w+') as log_file:
        for seq in os.listdir(spike_dir):
            if int(seq) < 84:
                continue
            if not os.path.exists(os.path.join(recon_dir, seq)):
                os.mkdir(os.path.join(recon_dir, seq))
            for sub_seq_file in os.listdir(os.path.join(spike_dir, seq)):
                if int(seq) == 84 and int(sub_seq_file.split('.')[0]) < 829:
                    continue
                s = time.time()
                im_index = random.randint(3, 5)
                save_path = os.path.join(recon_dir, seq, f'{sub_seq_file.split(".")[0]}_{im_index}.png')

                spk_voxel = np.load(os.path.join(spike_dir, seq, sub_seq_file))
                spk_voxel = spk_voxel[10 * (im_index - 3):10 * (im_index - 3) + 41]
                spk_voxel = torch.from_numpy(spk_voxel).float()
                spk_voxel = torch.unsqueeze(spk_voxel, 0)
                spk_voxel = spk_voxel.to(device)

                im_recon = model(spk_voxel)[0]
                im_recon = torch.squeeze(im_recon, 0)
                im_recon = torch.squeeze(im_recon, 0)
                im_recon = im_recon.cpu().detach().numpy()

                gt_path = os.path.join(gt_dir, seq, sub_seq_file.split(".")[0], f'im{im_index}.png')
                gt = Image.open(gt_path).convert('L')
                gt = np.array(gt) / 255
                mse_loss = mse(gt, im_recon)
                log_file.write(f'{seq}\{sub_seq_file.split(".")[0]}\im{im_index}.png\t{mse_loss}\n')
                log_file.flush()

                im_recon *= 255
                im_recon = Image.fromarray(im_recon)
                im_recon = im_recon.convert('L')
                im_recon.save(save_path)
                print(f'{seq}\{sub_seq_file.split(".")[0]}\im{im_index}.png\tTime cost:{round(time.time() - s, 2)}s')





