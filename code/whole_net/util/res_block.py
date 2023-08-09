from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.conv3(out)
        return self.relu2(x + out)