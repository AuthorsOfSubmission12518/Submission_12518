import torch
import torch.nn as nn

from whole_net.util.convlstm import ConvLSTM


def reverse_tensor_in_time(x):
    # B * T * C * H * W
    y = []
    for i in range(x.shape[1]):
        y.append(x[:, x.shape[1] - 1 - i, :, :, :])
    y = torch.stack(y, dim=1)
    return y


class convlstm_based_att(nn.Module):
    def __init__(self, c=4, k=8, layers=5, reverse=False):
        super(convlstm_based_att, self).__init__()

        self.c = c
        self.k = k
        self.layers = layers
        self.reverse = reverse
        self.convlstm = ConvLSTM(input_dim=c, hidden_dim=k, kernel_size=(3, 3), num_layers=layers, batch_first=True)

        self.conv0 = nn.Conv2d(in_channels=self.k, out_channels=32, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, padding=2, stride=2)

        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        _, T, _, _, _ = x.shape
        if not self.reverse:
            x_seg = x[:, : T // 2, :, :, :]
        else:
            x_seg = reverse_tensor_in_time(x[:, T // 2 + 1:, :, :, :])

        st_fea = self.convlstm(x_seg)
        st_fea = st_fea[:, -1, :, :, :]
        att_map = self.prelu(self.conv0(st_fea))
        att_map = self.sigmoid(self.conv1(att_map))

        return att_map


class bi_convlstm_based_att(nn.Module):
    def __init__(self, c=4, k=8, layers=5):
        super(bi_convlstm_based_att, self).__init__()

        self.c = c
        self.k = k
        self.layers = layers

        self.convlstm = ConvLSTM(input_dim=c, hidden_dim=k, kernel_size=(3, 3), num_layers=layers, batch_first=True)
        self.conv0 = nn.Conv2d(in_channels=2 * self.k, out_channels=32, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        _, T, _, _, _ = x.shape
        x_forawrd = x[:, : T // 2, :, :, :]
        x_backward = reverse_tensor_in_time(x[:, T // 2 + 1:, :, :, :])

        st_fea_forward = self.convlstm(x_forawrd)[:, -1, :, :, :]
        st_fea_backward = self.convlstm(x_backward)[:, -1, :, :, :]
        st_fea = torch.cat([st_fea_forward, st_fea_backward], dim=1)

        att_map = self.prelu0(self.conv0(st_fea))
        att_map = self.prelu1(self.conv1(att_map))
        att_map = self.sigmoid(self.conv2(att_map))

        return att_map


class bi_convlstm_based_att_with_refinement(nn.Module):
    def __init__(self, c=4, k=8, k1=16, layers=5):
        super(bi_convlstm_based_att_with_refinement, self).__init__()

        self.c = c
        self.k = k
        self.k1 = k1
        self.layers = layers

        self.convlstm = ConvLSTM(input_dim=c, hidden_dim=k, kernel_size=(3, 3), num_layers=layers, batch_first=True)
        self.conv0 = nn.Conv2d(in_channels=2 * self.k + k1, out_channels=self.k, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(in_channels=self.k, out_channels=self.k // 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.k // 2, out_channels=1, kernel_size=5, padding=2)

        self.refine0 = nn.Conv2d(1, k1, kernel_size=3, padding=1)
        self.refine1 = nn.Conv2d(k1, k1, kernel_size=3, padding=1)

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, recon_img):
        _, _, T, _, _ = x.shape
        x_forawrd = x[:, : T // 2, :, :, :]
        x_backward = reverse_tensor_in_time(x[:, T // 2 + 1:, :, :, :])

        st_fea_forward = self.convlstm(x_forawrd)[:, -1, :, :, :]
        st_fea_backward = self.convlstm(x_backward)[:, -1, :, :, :]
        st_fea = torch.cat([st_fea_forward, st_fea_backward], dim=1)

        refine_fea = self.relu(self.refine0(recon_img))
        refine_fea = self.relu(self.refine1(refine_fea))

        att_map = torch.cat([st_fea, refine_fea], dim=1)
        att_map = self.prelu0(self.conv0(att_map))
        att_map = self.prelu1(self.conv1(att_map))
        att_map = self.sigmoid(self.conv2(att_map))

        return att_map


if __name__ == '__main__':
    device = 'cuda:9'
    cl = bi_convlstm_based_att_with_refinement(4, 4, 3).to(device)
    import torch.optim as optim

    opti = optim.Adam(cl.parameters(), lr=1e-4)
    opti.zero_grad()

    x = torch.zeros(1, 27, 4, 256, 448)
    x = x.to(device)
    x1 = torch.zeros(1, 1, 256, 448)
    x1 = x1.to(device)
    y_hat = cl(x, x1)
    y = torch.zeros(1, 1, 256, 448)
    y = y.to(device)
    mse = nn.MSELoss()(y, y_hat)
    mse.backward()
    opti.step()
    print('..')



