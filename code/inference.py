# # import os
# # import json
# # import shutil
# # import sys
# # import time
# # import argparse
# #
# # import torch
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from torch.utils.tensorboard import SummaryWriter
# #
# # from whole_net.joint_net import Joint_Net
# # from com_net.util.loss import RateDistortionLoss
# #
# # class AverageMeter:
# #     def __init__(self):
# #         self.val = 0
# #         self.avg = 0
# #         self.sum = 0
# #         self.count = 0
# #
# #     def update(self, val, n=1):
# #         self.val = val
# #         self.sum += val * n
# #         self.count += n
# #         self.avg = self.sum / self.count
# #
# #     def reset(self):
# #         self.__init__()
# #
# #
# # def main():
# #     test_dataset = spk_provider(pms['spk_path'], pms['gt_path'], 'test')
# #     test_dataloader = DataLoader(
# #         test_dataset,
# #         batch_size=pms['test_batch'],
# #         num_workers=pms['worker_num'],
# #         shuffle=False,
# #         pin_memory=(pms['device'] != "cpu")
# #     )
# #
# #     net = Joint_Net(pms['recon_net_path'], pms['com_net_path'], pms['device'])
# #     net = net.to(pms['device'])
# #     net.recon_net.load_state_dict(torch.load(pms['recon_net_path'], map_location=pms['device'])['state_dict'])
# #     net.com_net.load_state_dict(torch.load(f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar', map_location=pms['device'])['state_dict'])
# #     criterion = RateDistortionLoss(pms['lmbda'])
# #
# #     net.eval()
# #     loss = AverageMeter()
# #     bpp_loss = AverageMeter()
# #     mse_loss = AverageMeter()
# #
# #     with torch.no_grad():
# #         for i, d in enumerate(test_dataloader):
# #             isi = d[0].to(pms['device'])
# #             img = d[1].to(pms['device'])
# #             out_net = net(isi)
# #             out_criterion = criterion(out_net, img)
# #
# #             loss.update(out_criterion["loss"].item())
# #             bpp_loss.update(out_criterion["bpp_loss"].item())
# #             mse_loss.update(out_criterion["mse_loss"].item())
# #
# #     print(
# #         f"Loss: {loss.avg:.6f} |"
# #         f"\tMSE loss: {mse_loss.avg:.6f} |"
# #         f"\tBpp loss: {bpp_loss.avg:.6f}"
# #     )
# #
# #     return loss.avg
# #
# #
# # parser = argparse.ArgumentParser(description='Reconstruction and Compression for spike sequences.')
# # parser.add_argument('--lmbda', type=int, required=True)
# # parser.add_argument('--device', type=str, required=True)
# # parser.add_argument('--test_batch_size', default=1, type=int)
# # args = parser.parse_args()
# #
# #
# # if __name__ == '__main__':
# #     from pms.joint.recon_cheng128 import pms
# #     from folder.spk_folder import spk_provider
# #
# #     print(f'current pid: {os.getpid()}')
# #     pms['lmbda'] = args.lmbda
# #     pms['device'] = args.device
# #     pms['test_batch'] = args.test_batch_size
# #     print(json.dumps(pms, indent=2))
# #     main()
#
#
# import os
# import json
# import shutil
# import sys
# import time
# import argparse
#
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# from whole_net.joint_net import Joint_Net
# from com_net.util.loss import RateDistortionLoss
#
# class AverageMeter:
#     def __init__(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#     def reset(self):
#         self.__init__()
#
#
# def main():
#     test_dataset = spk_provider(pms['spk_path'], pms['gt_path'], 'test')
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=pms['test_batch'],
#         num_workers=pms['worker_num'],
#         shuffle=False,
#         pin_memory=(pms['device'] != "cpu")
#     )
#
#     net = Joint_Net(pms['recon_net_path'], pms['com_net_path'], pms['device'])
#     net = net.to(pms['device'])
#     net.recon_net.load_state_dict(torch.load(pms['recon_net_path'], map_location=pms['device'])['state_dict'])
#     net.com_net.load_state_dict(torch.load(f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar', map_location=pms['device'])['state_dict'])
#     criterion = RateDistortionLoss(pms['lmbda'])
#
#     net.eval()
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     mse_loss = AverageMeter()
#
#     with torch.no_grad():
#         for i, d in enumerate(test_dataloader):
#             isi = d[0].to(pms['device'])
#             img = d[1].to(pms['device'])
#             out_net = net(isi)
#             out_criterion = criterion(out_net, img)
#
#             loss.update(out_criterion["loss"].item())
#             bpp_loss.update(out_criterion["bpp_loss"].item())
#             mse_loss.update(out_criterion["mse_loss"].item())
#
#     print(
#         f"Loss: {loss.avg:.6f} |"
#         f"\tMSE loss: {mse_loss.avg:.6f} |"
#         f"\tBpp loss: {bpp_loss.avg:.6f}"
#     )
#
#     return loss.avg
#
#
# parser = argparse.ArgumentParser(description='Reconstruction and Compression for spike sequences.')
# parser.add_argument('--lmbda', type=int, required=True)
# parser.add_argument('--device', type=str, required=True)
# parser.add_argument('--test_batch_size', default=1, type=int)
# args = parser.parse_args()
#
#
# if __name__ == '__main__':
#     from pms.joint.recon_cheng128 import pms
#     from folder.spk_folder import spk_provider
#
#     print(f'current pid: {os.getpid()}')
#     pms['lmbda'] = args.lmbda
#     pms['device'] = args.device
#     pms['test_batch'] = args.test_batch_size
#     print(json.dumps(pms, indent=2))
#     main()
#
import math

import numpy as np
import torch

from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att
from com_net.util.loss import RateDistortionLoss

from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage

lmbda = 0.0067

model_list = {
    'lic': {
            'model': Joint_Net_LIC(None, None, None),
            'checkpoint': f'/home/kxfeng/iccv/checkpoint/lic/joint-{lmbda}.pth'
        },
    'lstm': {
            'model': Joint_Net_LIC_with_lstm_att(None, None),
            'checkpoint': f'/home/kxfeng/iccv/checkpoint/lic/lstm_att/joint-{lmbda}.pth'
        },
    'bilstm': {
            'model': Joint_Net_LIC_with_bilstm_att(None, None),
            'checkpoint': f'/home/kxfeng/iccv/checkpoint/lic/bilstm_att/joint-{lmbda}.pth'
        },
}


def main():
    device = 'cuda:0'
    seq = '00036'
    sub_seq = '0606'
    im_idx = 4

    spk_path = f'/data/kxfeng/vimeo_septuplet_spike/{seq}/{sub_seq}.npy'
    gt_path = f'/data/klin/vimeo_septuplet/sequences/{seq}/{sub_seq}/im{im_idx}.png'

    for model in model_list:
        print(model)
        net = model_list[model]['model']
        net = net.to(device)
        net.load_state_dict(torch.load(model_list[model]['checkpoint'], map_location=device)['state_dict'])
        criterion = RateDistortionLoss()

        net.eval()

        with torch.no_grad():
            d_np = np.load(spk_path)
            d_ts = torch.from_numpy(d_np)
            d_ts = d_ts.float()
            d_ts = torch.squeeze(d_ts.to(device))
            d_ts = torch.unsqueeze(d_ts[10 * (im_idx - 3):10 * (im_idx - 3) + 41], 0)
            d_ts = CenterCrop((256, 256))(d_ts)

            gt_img = Image.open(gt_path).convert('L')
            gt_ts = ToTensor()(gt_img)
            gt_ts = torch.unsqueeze(gt_ts.to(device), 0)
            gt_ts = CenterCrop((256, 256))(gt_ts)

            out_net = net(d_ts)
            img = ToPILImage()(torch.squeeze(out_net['x_hat']))
            img.save(f'/home/kxfeng/{model}.png')

            out_criterion = criterion(out_net, gt_ts)
            print(out_criterion['bpp_loss'].item())
            print(out_criterion['mse_loss'].item())
            print(-10 * math.log10(out_criterion['mse_loss'].item()))



if __name__ == '__main__':
    main()

