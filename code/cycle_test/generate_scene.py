import math
import os.path
import time

import numpy as np
import torch
from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att
from com_net.util.loss import RateDistortionLoss

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import shutil


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
    metric = RateDistortionLoss()
    to_ts = ToTensor()
    to_pil = ToPILImage()
    sample_num = 10
    cur_sample_num = 0

    # for seq in os.listdir(root_dir):
    #     if cur_sample_num > sample_num:
    #         break
    #     for sub_seq in os.listdir(os.path.join(root_dir, seq)):
    #         cur_sample_num += 1
    #         if cur_sample_num > sample_num:
    #             break
    #         shutil.rmtree(os.path.join(root_dir, seq, sub_seq, 'recon_frames'))


    for model_name in model_list:
        model = model_list[model_name]['model']
        model = model.to(device)
        for lmbda in lmbda_list:
            print(model_name, lmbda)
            # if model_name == 'lic' and lmbda in [0.05, 0.025, 0.013]:
            #     continue
            checkpoint_path = os.path.join(model_list[model_name]['checkpoint'], f'joint-{lmbda}.pth')
            checkpoint = torch.load(checkpoint_path, map_location=device)['state_dict']
            model.load_state_dict(checkpoint)
            model.eval()

            bpp_ave = AverageMeter()
            mse_ave = AverageMeter()
            cur_sample_num = 0

            with torch.no_grad():
                for seq in os.listdir(root_dir):
                    if cur_sample_num > sample_num:
                        break
                    for sub_seq in os.listdir(os.path.join(root_dir, seq)):
                        cur_sample_num += 1
                        if cur_sample_num > sample_num:
                            break
                        if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 'recon_frames')):
                            os.mkdir(os.path.join(root_dir, seq, sub_seq, 'recon_frames'))
                        if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name)):
                            os.mkdir(os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name))
                        if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name, str(lmbda))):
                            os.mkdir(os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name, str(lmbda)))

                        spk_voxel = np.load(os.path.join(root_dir, seq, sub_seq, 'spk.npy'))
                        s = time.time()
                        for i in range(spk_voxel.shape[0] - 40):
                            spk_block = spk_voxel[i: i + 41, :, 96: -96]
                            spk_block = torch.from_numpy(spk_block).float()
                            spk_block = torch.unsqueeze(spk_block, 0)
                            spk_block = spk_block.to(device)

                            img = Image.open(os.path.join(root_dir, seq, sub_seq, 'frames', f'{i + 21}.png'))
                            img = to_ts(img)
                            img = torch.unsqueeze(img[:, :, 96: -96], 0)
                            img = img.to(device)

                            out_net = model(spk_block)
                            loss = metric(out_net, img)

                            bpp_ave.update(loss['bpp_loss'].item())
                            mse_ave.update(loss['mse_loss'].item())

                            rec_img = out_net['x_hat']
                            rec_img = torch.squeeze(rec_img)
                            rec_img = to_pil(rec_img)
                            rec_img.save(os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name, str(lmbda), f'{i + 21}.png'))
                        print(seq, sub_seq, round(time.time() - s, 2))

            print(model_name, lmbda, bpp_ave.avg, mse_ave.avg, -10 * math.log10(mse_ave.avg))


# if __name__ == '__main__':
#     model_list = {
#         'lic': {
#             'model': Joint_Net_LIC(None, None, None),
#             'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\'
#         },
#         'lstm': {
#             'model': Joint_Net_LIC_with_lstm_att(None, None),
#             'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\soam\\'
#         },
#         'bilstm': {
#             'model': Joint_Net_LIC_with_bilstm_att(None, None),
#             'checkpoint': f'C:\\Users\\fengk\\Desktop\\checkpoint\\bisoam\\'
#         },
#     }
#
#     lmbda_list = [0.05, 0.025, 0.013, 0.0067]
#
#     root_dir = 'G:\\vimeo_septuplet\\vimeo_septuplet_spike_91'
#
#     for model_name in model_list:
#         for lmbda in lmbda_list:
#             for seq in ['00001']:
#                 for sub_seq in [
#                     '0001',
#                     '0002',
#                     '0003',
#                     '0004',
#                     '0005',
#                     '0006',
#                     '0007',
#                     '0008',
#                     '0009',
#                     '0010',
#                 ]:
#                     if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 'recon_spk')):
#                         os.mkdir(os.path.join(root_dir, seq, sub_seq, 'recon_spk'))
#                     recon_dir = os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name, str(lmbda))
#
#                     save_path = os.path.join(root_dir, seq, sub_seq, 'recon_spk', f'{model_name}-{lmbda}.npy')
#                     threshold = 255
#                     light_scale = 128
#                     integrator = np.random.random((256, 256)) * threshold
#                     light_intensity = light_scale / 256
#                     spk_list = []
#
#                     for i in range(len(os.listdir(recon_dir))):
#                         recon_path = os.path.join(recon_dir, f'{i + 21}.png')
#                         recon_img = Image.open(recon_path)
#                         recon_img_np = np.array(recon_img)
#                         integrator += recon_img_np * light_intensity
#                         spk = integrator >= threshold
#                         spk_list.append(spk)
#                         integrator -= spk * 255
#
#                     spk_list = np.array(spk_list, dtype='uint8')
#                     np.save(save_path, spk_list)
#                     print(save_path)

