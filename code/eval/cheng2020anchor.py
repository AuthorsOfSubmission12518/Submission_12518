import os.path
import time

import torch

from torch.utils.data import DataLoader

from compressai.models.waseda import *
from compressai.models.google import *
from compressai.zoo import *
from eval.util import RateDistortionLoss, download_and_extract
from eval.spk_3ch import img_folder
from eval.params import model_urls, cfgs


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


if __name__ == '__main__':
    device = 'cuda:3'
    model_name_list = {
        'bmshj2018-factorized': [bmshj2018_factorized, FactorizedPrior, 8],
        'bmshj2018-hyperprior': [bmshj2018_hyperprior, ScaleHyperprior, 8],
        'mbt2018-mean': [mbt2018_mean, MeanScaleHyperprior, 8],
        'mbt2018': [mbt2018, JointAutoregressiveHierarchicalPriors, 8],
        'cheng2020-attn': [cheng2020_attn, Cheng2020Attention, 6],
        'cheng2020-anchor': [cheng2020_anchor, Cheng2020Anchor, 6]
    }
    data_dir = '/data2/kxfeng/vimeo_septuplet_recon/raw_rgb/'
    gt_dir = '/backup1/klin/vimeo_septuplet/sequences'
    # checkpoint_dir = '/home/kxfeng/checkpoint'
    # if not os.path.exists(checkpoint_dir):
    #     os.mkdir(checkpoint_dir)
    # checkpoint_dir = f'/home/kxfeng/checkpoint/{model_name}'
    # if not os.path.exists(checkpoint_dir):
    #     os.mkdir(checkpoint_dir)


    test_dataset = img_folder(data_dir, gt_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device != "cpu")
    )


    # url_list = model_urls[model_name]['mse']
    # param_list = cfgs[model_name]
    # for k in url_list:
    #     v = url_list[k]
    #     download_and_extract(v, checkpoint_dir)
    for model_name in model_name_list:
        for q in range(1, model_name_list[model_name][2] + 1):
            model = model_name_list[model_name][0](quality=q, pretrained=True)
            model_name_list[model_name][1].from_state_dict(model.state_dict())
            model = model.to(device)
            print(model_name, q)

            model.eval()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            ssim_loss = AverageMeter()
            ms_ssim_loss = AverageMeter()
            time_avg = AverageMeter()
            criterion = RateDistortionLoss()
            with torch.no_grad():
                for i, d in enumerate(test_dataloader):
                    input = d[0].to(device)
                    gt = d[1].to(device)
                    gt1 = input[:, 0: 1, :, :]
                    s = time.time()
                    out_net = model(input)

                    out_criterion = criterion(out_net, gt1, device)

                    bpp_loss.update(out_criterion["bpp_loss"])
                    mse_loss.update(out_criterion["mse_loss"])
                    ssim_loss.update(out_criterion["ssim_loss"])
                    ms_ssim_loss.update(out_criterion["msssim_loss"])
                    time_avg.update(time.time() - s)
                    # print(f'Finish index {i}.')

                print(
                    f"Average losses:"
                    f"\tMSE loss: {mse_loss.avg:.6f} |"
                    f"\tBpp loss: {bpp_loss.avg:.6f} |"
                    f"\tSSIM loss: {ssim_loss.avg:.6f} |"
                    f"\tMS-SSIM loss: {ms_ssim_loss.avg:.6f} |"
                    f"\tTime: {time_avg.avg:.6f} |"
                )

