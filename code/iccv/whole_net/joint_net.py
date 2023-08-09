import torch
import torch.nn as nn

from cheng2020 import Cheng2020
from LIC import TCM
from recon_net.spk2imgnet import new_Spike_Net_v3 as recon_net
from whole_net.lstm_based_attention import convlstm_based_att, bi_convlstm_based_att
from whole_net.conv_based_attention import conv2d_based_att, conv3d_based_att


class Joint_Net(nn.Module):
    def __init__(self, recon_net_path, com_net_path, device):
        super(Joint_Net, self).__init__()

        self.recon_net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
        self.com_net = Cheng2020(128)

        # recon_net_type = recon_net_path.split('/')[-1]
        # if 'small' in recon_net_type:
        #     from recon_net.spk2imgnet_small import new_Spike_Net_v3 as recon_net
        # elif 'shortterm' in recon_net_type:
        #     from recon_net.spk2imgnet_shortterm import new_Spike_Net_v3 as recon_net
        # else:
        #     from recon_net.spk2imgnet import new_Spike_Net_v3 as recon_net
        # self.recon_net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
        #
        # com_net_type = com_net_path.split('/')[-1]
        # if '64' in com_net_type:
        #     self.com_net = Cheng2020(64)
        # else:
        #     self.com_net = Cheng2020(128)
        #
        # self.recon_net.load_state_dict(torch.load(recon_net_path, map_location=device)['state_dict'])
        # self.com_net.load_state_dict(torch.load(com_net_path, map_location=device)['state_dict'])

    def forward(self, x):
        y = self.recon_net(x)[0]
        out = self.com_net(y)
        out['x_hat1'] = y
        return out


class Joint_Net_with_lstm_att(Joint_Net):
    def __init__(self, joint_net_path, device):
        super(Joint_Net_with_lstm_att, self).__init__(None, None, None)

        # self.load_state_dict(torch.load(joint_net_path, map_location=device)['state_dict'])
        # print(f'Load from {joint_net_path}')

        self.temporal_att = convlstm_based_att(c=1, k=4, layers=4)
        self.first_encoder_layer = self.com_net.g_a[0]
        self.com_net.g_a = self.com_net.g_a[1:]

    def forward(self, x):
        y = self.recon_net(x)[0]
        temp_att_map = self.temporal_att(x)
        y_fea = self.first_encoder_layer(y)
        y_fea = y_fea + y_fea * temp_att_map
        out = self.com_net(y_fea)
        out['x_hat1'] = y
        return out


class Joint_Net_with_conv_att(Joint_Net_with_lstm_att):
    def __init__(self, joint_net_path, device):
        super().__init__(joint_net_path, device)

        self.temporal_att = conv2d_based_att(in_ch=41, mid_ch=16, out_ch=128, res_num=8)


class Joint_Net_LIC(nn.Module):
    def __init__(self, recon_net_path, com_net_path, device):
        super(Joint_Net_LIC, self).__init__()
        self.recon_net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
        self.com_net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)

        # self.recon_net.load_state_dict(torch.load(recon_net_path, map_location=device)['state_dict'])
        # self.com_net.load_state_dict(torch.load(com_net_path, map_location=device)['state_dict'])
        # print('Loading from two pretrained models.')

    def forward(self, x):
        y = self.recon_net(x)[0]
        out = self.com_net(y)
        out['x_hat1'] = y
        return out


class Joint_Net_LIC_with_lstm_att(Joint_Net_LIC):
    def __init__(self, joint_net_path, device):
        super(Joint_Net_LIC_with_lstm_att, self).__init__(None, None, None)

        # self.load_state_dict(torch.load(joint_net_path, map_location=device)['state_dict'])
        # print(f'Load from {joint_net_path}')

        self.temporal_att = convlstm_based_att(c=1, k=4, layers=4)
        self.first_encoder_layer = self.com_net.g_a[0]
        self.com_net.g_a = self.com_net.g_a[1:]

        # for m in self.temporal_att.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.recon_net(x)[0]
        temp_att_map = self.temporal_att(x)
        y_fea = self.first_encoder_layer(y)
        y_fea = y_fea + y_fea * temp_att_map
        out = self.com_net(y_fea)
        out['x_hat1'] = y
        out['att_map'] = temp_att_map
        return out


class Joint_Net_LIC_with_bilstm_att(Joint_Net_LIC):
    def __init__(self, joint_net_path, device):
        super(Joint_Net_LIC_with_bilstm_att, self).__init__(None, None, None)

        # self.load_state_dict(torch.load(joint_net_path, map_location=device)['state_dict'])
        # print(f'Load from {joint_net_path}')

        self.temporal_att = bi_convlstm_based_att(c=1, k=4, layers=4)
        self.first_encoder_layer = self.com_net.g_a[0]
        self.com_net.g_a = self.com_net.g_a[1:]

        # for m in self.temporal_att.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.recon_net(x)[0]
        temp_att_map = self.temporal_att(x)
        y_fea = self.first_encoder_layer(y)
        y_fea = y_fea + y_fea * temp_att_map
        out = self.com_net(y_fea)
        out['x_hat1'] = y
        out['att_map'] = temp_att_map
        return out