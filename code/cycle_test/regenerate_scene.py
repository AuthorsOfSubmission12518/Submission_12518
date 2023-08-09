import os

import numpy as np
import torch
from PIL import Image

from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att
from recon_net.spk2imgnet import new_Spike_Net_v3

from torchvision.transforms import ToPILImage

if __name__ == '__main__':
    model_list = {
        'lic': {
            'model': Joint_Net_LIC(None, None, None),
            'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\'
        },
        'lstm': {
            'model': Joint_Net_LIC_with_lstm_att(None, None),
            'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\soam\\'
        },
        'bilstm': {
            'model': Joint_Net_LIC_with_bilstm_att(None, None),
            'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\bisoam\\'
        },
    }

    lmbda_list = [0.05, 0.025, 0.013, 0.0067]

    root_dir = 'G:\\vimeo_septuplet\\vimeo_septuplet_spike_91'
    device = 'cuda:0'

    recon_net = new_Spike_Net_v3(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    checkpoint_path = 'C:\\Users\\fengk\\Desktop\\checkpoint\\recon_net.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)['state_dict']
    recon_net.load_state_dict(checkpoint)
    recon_net = recon_net.to(device)

    to_pil = ToPILImage()

    # for model_name in model_list:
    #     for lmbda in lmbda_list:
    #         for seq in ['00001']:
    #             for sub_seq in [
    #                 '0001',
    #                 '0002',
    #                 '0003',
    #                 '0004',
    #                 '0005',
    #                 '0006',
    #                 '0007',
    #                 '0008',
    #                 '0009',
    #                 '0010',
    #             ]:
    #                 if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 're_recon_scene')):
    #                     os.mkdir(os.path.join(root_dir, seq, sub_seq, 're_recon_scene'))
    #                 spk_dir = os.path.join(root_dir, seq, sub_seq, 'recon_spk')
    #                 spk_path = os.path.join(spk_dir, f'{model_name}-{lmbda}.npy')
    #                 spk_voxel = np.load(spk_path)
    #                 spk_voxel = torch.from_numpy(spk_voxel[10: ]).float()
    #                 spk_voxel = torch.unsqueeze(spk_voxel, 0)
    #                 spk_voxel = spk_voxel.to(device)
    #
    #                 re_recon_img = recon_net(spk_voxel)[0]
    #                 re_recon_img = torch.clip(re_recon_img, 0, 1)
    #                 re_recon_img = torch.squeeze(re_recon_img)
    #                 re_recon_img = to_pil(re_recon_img)
    #                 save_path = os.path.join(root_dir, seq, sub_seq, 're_recon_scene', f'{model_name}-{lmbda}.png')
    #                 re_recon_img.save(save_path)

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
            spk_path = os.path.join(root_dir, seq, sub_seq, 'spk.npy')
            spk_voxel = np.load(spk_path)
            spk_voxel = torch.from_numpy(spk_voxel[30: 71, :, 96: -96]).float()
            spk_voxel = torch.unsqueeze(spk_voxel, 0)
            spk_voxel = spk_voxel.to(device)

            base_img = recon_net(spk_voxel)[0]
            base_img = torch.clip(base_img, 0, 1)
            base_img = torch.squeeze(base_img)
            base_img = to_pil(base_img)
            save_path = os.path.join(root_dir, seq, sub_seq, 'base.png')
            base_img.save(save_path)
