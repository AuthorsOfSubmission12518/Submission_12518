import torch.nn as nn

from recon_net.extractor import Feature_Extractor


class new_Spike_Net_v3(nn.Module):
    def __init__(self, in_channels, features, out_channels, win_r, win_step):
        super(new_Spike_Net_v3, self).__init__()
        self.extractor = Feature_Extractor(in_channels=in_channels, features=features, out_channels=features,
                                           channel_step=1, num_of_layers=12)

        self.rec_conv0 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)
        self.rec_conv1 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=3, padding=1)
        self.rec_relu = nn.ReLU()

    def forward(self, x):
        out, est = self.extractor(x)

        out = self.rec_relu(self.rec_conv0(out))
        out = self.rec_conv1(out)

        return out, est
