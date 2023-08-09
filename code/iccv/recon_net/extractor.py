import torch
import torch.nn as nn


class CALayer_v2(nn.Module):
    def __init__(self, in_channels):
        super(CALayer_v2, self).__init__()
        self.ca_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        weight = self.ca_block(x)
        return weight


class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return self.relu3(x + out)


class Feature_Extractor(nn.Module):
    def __init__(self, in_channels, features, out_channels, channel_step, num_of_layers=16):
        super(Feature_Extractor, self).__init__()
        # self.InferLayer = LightInferLayer(in_channels=in_channels)
        self.channel_step = channel_step
        self.conv0_0 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=in_channels - 2 * channel_step, out_channels=16, kernel_size=3, padding=1)
        self.conv0_2 = nn.Conv2d(in_channels=in_channels - 4 * channel_step, out_channels=16, kernel_size=3, padding=1)
        self.conv0_3 = nn.Conv2d(in_channels=in_channels - 6 * channel_step, out_channels=16, kernel_size=3, padding=1)

        self.conv1_0 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

        self.ca = CALayer_v2(in_channels=4)
        self.conv = nn.Conv2d(in_channels=4, out_channels=features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(BasicBlock(features=features))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out_0 = self.conv1_0(self.relu(self.conv0_0(x)))
        out_1 = self.conv1_1(self.relu(self.conv0_1(x[:, self.channel_step:-self.channel_step, :, :])))
        out_2 = self.conv1_2(self.relu(self.conv0_2(x[:, 2 * self.channel_step:-2 * self.channel_step, :, :])))
        out_3 = self.conv1_3(self.relu(self.conv0_3(x[:, 3 * self.channel_step:-3 * self.channel_step, :, :])))

        out = torch.cat((out_0, out_1), 1)
        out = torch.cat((out, out_2), 1)
        out = torch.cat((out, out_3), 1)

        est = out
        weight = self.ca(out)

        out = weight * out
        out = self.conv(out)
        out = self.relu(out)
        tmp = out
        out = self.net(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.conv3(out)
        return out + tmp, est


if __name__ == '__main__':
    extractor = Feature_Extractor(in_channels=13, features=64, out_channels=13, channel_step=1, num_of_layers=12)
    extractor.load_state_dict(torch.load('C:\\Users\\fengk\\Desktop\\extractor.pth', map_location='cuda:0'))
