import os

import numpy as np
from PIL import Image

from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att

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

    for model_name in model_list:
        for lmbda in lmbda_list:
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
                    if not os.path.exists(os.path.join(root_dir, seq, sub_seq, 'recon_spk')):
                        os.mkdir(os.path.join(root_dir, seq, sub_seq, 'recon_spk'))
                    recon_dir = os.path.join(root_dir, seq, sub_seq, 'recon_frames', model_name, str(lmbda))
                    try:
                        if len(os.listdir(recon_dir)) < 51:
                            print(f'Pass {recon_dir}')
                            continue
                    except FileNotFoundError:
                        print(f'Pass {recon_dir}')
                        continue

                    save_path = os.path.join(root_dir, seq, sub_seq, 'recon_spk', f'{model_name}-{lmbda}.npy')
                    if os.path.exists(save_path):
                        print(f'Already complete {recon_dir}')
                        continue

                    threshold = 255
                    light_scale = 128
                    integrator = np.random.random((256, 256)) * threshold
                    light_intensity = light_scale / 256
                    spk_list = []

                    for i in range(len(os.listdir(recon_dir))):
                        recon_path = os.path.join(recon_dir, f'{i + 21}.png')
                        recon_img = Image.open(recon_path)
                        recon_img_np = np.array(recon_img)
                        integrator += recon_img_np * light_intensity
                        spk = integrator >= threshold
                        spk_list.append(spk)
                        integrator -= spk * 255

                    spk_list = np.array(spk_list, dtype='uint8')
                    np.save(save_path, spk_list)
                    print(save_path)