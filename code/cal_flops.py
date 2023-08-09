import torch
from thop import profile, clever_format

from whole_net.joint_net import Joint_Net_LIC, Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att

model = Joint_Net_LIC_with_bilstm_att(None, None)

net = model.temporal_att
inputs = torch.randn(1, 41, 256, 256)
flops, total_params = profile(net, (inputs,))
macs, params = clever_format([flops, total_params], "%.3f")
print('flops: ', flops, 'params: ', total_params, params, 'macs: ', macs)

# net = model.recon_net.rec_conv1
# inputs = torch.randn(1, 64 * 3, 256, 256)
# flops, total_params = profile(net, (inputs,))
# macs, params = clever_format([flops, total_params], "%.3f")
# print('flops: ', flops, 'params: ', total_params, params, 'macs: ', macs)
#
# net = model.recon_net.rec_conv2
# inputs = torch.randn(1, 64, 256, 256)
# flops, total_params = profile(net, (inputs,))
# macs, params = clever_format([flops, total_params], "%.3f")
# print('flops: ', flops, 'params: ', total_params, params, 'macs: ', macs)


# model = Joint_Net_LIC(None, None, None)
# inputs = torch.randn(1, 41, 256, 256)
# with torch.no_grad():
#     model(inputs)