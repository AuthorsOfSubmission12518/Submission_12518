# import os
# import json
# import shutil
# import sys
# import time
# import argparse
#
# import math
# import torch
# import torch.optim as optim
# from torch import nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# from whole_net.joint_net import Joint_Net_with_lstm_att, Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, \
#     Joint_Net_LIC_with_bilstm_att
# from com_net.util.loss import RateDistortionLoss, RDDLoss
#
# from pms.joint.recon_cheng128 import pms
# from folder.spk_folder import spk_provider
#
# def test_one_epoch(
#         model,
#         test_dataloader):
#     model.eval()
#
#     with torch.no_grad():
#         for i, d in enumerate(test_dataloader):
#             isi = d[0].to(pms['device'])
#             img = d[1].to(pms['device'])
#             out_net = model(isi)
#             from torchvision.transforms import ToPILImage
#             img = ToPILImage()(torch.squeeze(out_net['x_hat']))
#             img.save(f'/data/kxfeng/recon_seq/{pms["lmbda"]}/{d[2]}.png')
#
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
#     # net = Joint_Net_LIC_with_bilstm_att(pms['load_checkpoint'], pms['device'])
#     net = Joint_Net_LIC(pms['load_checkpoint'], pms['device'], None)
#     if pms['recon_detach']:
#         for p in net.recon_net.parameters():
#             p.requires_grad_(False)
#     if pms['com_detach']:
#         for p in net.com_net.parameters():
#             p.requires_grad_(False)
#         for p in net.first_encoder_layer.parameters():
#             p.requires_grad_(False)
#     net = net.to(pms['device'])
#
#     last_epoch = 0
#     if pms['load_checkpoint']:
#         try:
#             checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
#             last_epoch = checkpoint["epoch"] + 1
#             print("Loading from ", pms['load_checkpoint'])
#             net.load_state_dict(checkpoint["state_dict"])
#             print("Loading finish")
#         except FileNotFoundError:
#             print("Loading failed")
#     else:
#         print("Loading from two pre-trained models.")
#         net.recon_net.load_state_dict(torch.load(pms['recon_net_path'], map_location=pms['device'])['state_dict'])
#         net.com_net.load_state_dict(
#             torch.load(f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar',
#                        map_location=pms['device'])['state_dict'])
#         print("Loading finish")
#
#     test_one_epoch(
#         net,
#         test_dataloader)
#
#
# if __name__ == '__main__':
#     print(f'current pid: {os.getpid()}')
#     pms['lmbda'] = 0.05
#     pms['device'] = 'cuda:0'
#     pms['batch'] = 2
#     pms['test_batch'] = 1
#     pms['worker_num'] = 1
#     pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/lic/joint-{pms["lmbda"]}.pth'
#     pms['rdd'] = False
#     pms['recon_detach'] = False
#     pms['com_detach'] = False
#     pms['save_checkpoint'] = '/home/kxfeng/iccv/checkpoint/lic/bilstm_att'
#     pms['learning_rate'] = 1e-6
#     print(json.dumps(pms, indent=2))
#     main()

# import cv2
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def visulize_attention_ratio(attention_mask, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    # print("load image from: ", img_path)
    # # load the image
    # img = Image.open(img_path, mode='r')
    # img_h, img_w = img.size[0], img.size[1]
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    #
    # # scale the image
    # img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    # img = img.resize((img_h, img_w))
    # plt.imshow(img, alpha=1)
    # plt.axis('off')

    # normalize the attention mask
    # mask = cv2.resize(attention_mask, (img_h, img_w))
    mask = attention_mask
    # normed_mask = mask / mask.max()
    # normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    # plt.show()


if __name__ == '__main__':
    att_map = np.zeros((128, 128))
    visulize_attention_ratio(attention_mask=att_map)
