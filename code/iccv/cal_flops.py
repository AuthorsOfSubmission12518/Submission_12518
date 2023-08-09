import torch
from thop import profile, clever_format

from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att


inputs = torch.randn(1, 41, 256, 256)

for net in [
    Joint_Net_LIC(None, None, None),
    Joint_Net_LIC_with_lstm_att(None, None),
    Joint_Net_LIC_with_bilstm_att(None, None)
]:
    flops, total_params = profile(net, (inputs,))
    macs, params = clever_format([flops, total_params], "%.3f")
    print('flops: ', flops, 'params: ', total_params, params, 'macs: ', macs)