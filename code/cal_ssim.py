import os
import numpy as np
import math
import torch
import torchvision.transforms
from pytorch_msssim import ms_ssim, ssim
from torch.utils.data import DataLoader

from cheng2020 import AverageMeter
from folder.spk_folder import spk_provider, spk_provider_real
from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att


def visulize_attention_ratio(attention_mask, cmap="jet"):
    # print(attention_mask)
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    import torch.nn as nn
    # attention_mask = nn.Sigmoid()(attention_mask)

    mask = np.array(attention_mask.cpu())
    mask = np.log(mask)
    # mask = np.clip(mask, 1e-8, 1e-6)
    normed_mask = mask / mask.max()
    # normed_mask = np.resize(normed_mask, (128 * 128,))
    # mask = sorted(normed_mask)
    # print(mask)
    normed_mask = (normed_mask * 255).astype('uint8')
    # print(np.average(normed_mask))
    # print(np.var(normed_mask))

    plt.imshow(normed_mask, interpolation='nearest', cmap='coolwarm')
    # plt.show()
    plt.savefig(f'/home/kxfeng/att_map/{len(os.listdir("/home/kxfeng/att_map/"))}.jpg')
    print(len(os.listdir("/home/kxfeng/att_map/")))


if __name__ == '__main__':
    # spk_path = '/data/kxfeng/vimeo_septuplet_spike'
    # gt_path = '/data/klin/vimeo_septuplet/sequences'
    spk_path = '/home/kxfeng/M7398_voxel/'
    save_path = '/home/kxfeng/M7398_voxel_recon/'
    device = 'cuda:7'
    to_pil = torchvision.transforms.ToPILImage()

    test_dataset = spk_provider_real(spk_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=(device != "cpu")
    )

    model_list = {
        # '/home/kxfeng/iccv/checkpoint/lic/': Joint_Net_LIC(None, None, None),
        '/home/kxfeng/iccv/checkpoint/lic/lstm_att/': Joint_Net_LIC_with_lstm_att(None, None),
        '/home/kxfeng/iccv/checkpoint/lic/bilstm_att/': Joint_Net_LIC_with_bilstm_att(None, None)
    }
    ave = {}
    # for seq_name in ['car', 'cpl1', 'rotation1', 'rotation2', 'train', 'ballon']:
    for seq_name in ['ballon']:
        ave[seq_name] = {}
        # for model_name in ['lic', 'lstm_att', 'bilstm_att']:
        for model_name in ['lstm_att', 'bilstm_att']:
            ave[seq_name][model_name] = {}
            for lmbda in [0.05, 0.025, 0.013, 0.0067]:
                ave[seq_name][model_name][lmbda] = AverageMeter()
    for checkpoint_dir in model_list:
        # print(checkpoint_dir)
        model = model_list[checkpoint_dir]
        model_name = checkpoint_dir.split('/')[-2]
        if not os.path.exists(os.path.join(save_path, model_name)):
            os.mkdir(os.path.join(save_path, model_name))
        model.to(device)
        for lmbda in [.05]:
            if not os.path.exists(os.path.join(save_path, model_name, str(lmbda))):
                os.mkdir(os.path.join(save_path, model_name, str(lmbda)))
            checkpoint_path = os.path.join(checkpoint_dir, f'joint-{lmbda}.pth')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

            model.eval()
            ssim_ave = AverageMeter()
            msssim_ave = AverageMeter()

            with torch.no_grad():
                for i, d in enumerate(test_dataloader):
                    file_name = d[0][0].split('.')[0]
                    spk_voxel = d[1].to(device)
                    out_net = model(spk_voxel)
                    # visulize_attention_ratio(out_net['att_map'][0][0])
                    for j in range(128):
                        visulize_attention_ratio(out_net['att_map'][0][j])
                    out_img = to_pil(torch.squeeze(torch.clamp(out_net['x_hat'], 0, 1)))
                    # out_img.save(os.path.join(save_path, model_name, str(lmbda), f'{file_name}.png'))
                    seq_name = file_name.split('-')[0]
                    # print(model_name, lmbda,
                    # , sum(
                    #     (torch.log(likelihoods).sum() / (-math.log(2) * (256 * 256)))
                    #     for likelihoods in out_net["likelihoods"].values()
                    # ).item())
                    ave[seq_name][model_name][lmbda].update(sum(
                        (torch.log(likelihoods).sum() / (-math.log(2) * (256 * 256)))
                        for likelihoods in out_net["likelihoods"].values()
                    ).item())
                    # ssim_ave.update(ssim(out_net['x_hat'], img, data_range=1))
                    # msssim_ave.update(ms_ssim(out_net['x_hat'], img, data_range=1))

            # print(lmbda)
            # print(f'SSIM: {ssim_ave.avg}')
            # print(f'MS-SSIM: {msssim_ave.avg}')
    for seq_name in ['car', 'cpl1', 'rotation1', 'rotation2', 'train', 'ballon']:
        for model_name in ['lic', 'lstm_att', 'bilstm_att']:
            for lmbda in [0.05, 0.025, 0.013, 0.0067]:
                print(seq_name, model_name, lmbda, ave[seq_name][model_name][lmbda].avg)

