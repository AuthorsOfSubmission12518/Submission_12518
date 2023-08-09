import torch
import torch.nn as nn

from whole_net.util.res_block import BasicBlock


class conv2d_based_att(nn.Module):
    def __init__(self, in_ch=41, mid_ch=16, out_ch=128, res_num=8):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        layers0 = []
        for _ in range(res_num // 2):
            layers0.append(BasicBlock(features=mid_ch))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.res_net0 = nn.Sequential(*layers0)
        layers1 = []
        for _ in range(res_num // 2):
            layers1.append(BasicBlock(features=mid_ch * 2))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.res_net1 = nn.Sequential(*layers1)

        self.first_layer = nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, stride=1, kernel_size=3, padding=1)
        self.down_sample = nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch * 2, stride=2, kernel_size=3, padding=1)
        self.last_layer = nn.Conv2d(in_channels=mid_ch * 2, out_channels=out_ch, stride=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fea = self.relu(self.first_layer(x))
        fea = self.res_net0(fea)
        fea = self.relu(self.down_sample(fea))
        fea = self.res_net1(fea)
        att_map = self.sigmoid(self.last_layer(fea))

        return att_map


class conv3d_based_att(nn.Module):
    def __init__(self, c=4, k=8, N=64):
        super(conv3d_based_att, self).__init__()
        self.c = c
        self.k = k
        self.N = N

        self.conv00 = nn.Conv3d(in_channels=c, out_channels=k, stride=1, kernel_size=3, padding=1)
        self.conv01 = nn.Conv3d(in_channels=k, out_channels=k, stride=1, kernel_size=3, padding=1)
        self.conv02 = nn.Conv3d(in_channels=k, out_channels=k, stride=1, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(5 * k, N, kernel_size=5, padding=2)
        self.conv11 = nn.Conv2d(N, 1, kernel_size=3, padding=1)

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea0, fea1, fea2, fea3, fea4):
        B, _, H, W = fea0.shape()

        fea_seq = torch.stack([fea0, fea1, fea2, fea3, fea4], 2)
        st_fea = self.prelu0(self.conv00(fea_seq))
        st_fea = self.prelu1(self.conv01(st_fea))
        st_fea = self.prelu2(self.conv02(st_fea))

        att_map = st_fea.view(B, 5 * self.k, H, W)
        att_map = self.relu(self.conv10(att_map))
        att_map = self.sigmoid(self.conv11(att_map))

        return att_map


class conv3d_based_att_with_refinement(nn.Module):
    def __init__(self, c=4, k=8, k1=16, N=64):
        super(conv3d_based_att_with_refinement, self).__init__()

        self.c = c
        self.k = k
        self.k1 = k1
        self.N = N

        self.conv00 = nn.Conv3d(in_channels=c, out_channels=k, stride=1, kernel_size=3, padding=1)
        self.conv01 = nn.Conv3d(in_channels=k, out_channels=k, stride=1, kernel_size=3, padding=1)
        self.conv02 = nn.Conv3d(in_channels=k, out_channels=k, stride=1, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(5 * k + k1, N, kernel_size=5, padding=2)
        self.conv11 = nn.Conv2d(N, 1, kernel_size=3, padding=1)

        self.refine0 = nn.Conv2d(1, k1, kernel_size=3, padding=1)
        self.refine1 = nn.Conv2d(k1, k1, kernel_size=3, padding=1)

        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea_seq, recon_img):
        # B * C * T * H * W
        B, _, _, H, W = fea_seq.shape()

        st_fea = self.prelu0(self.conv00(fea_seq))
        st_fea = self.prelu1(self.conv01(st_fea))
        st_fea = self.prelu2(self.conv02(st_fea))
        st_fea = st_fea.view(B, 5 * self.k, H, W)

        refine_fea = self.relu(self.refine0(recon_img))
        refine_fea = self.relu(self.refine1(refine_fea))

        att_map = torch.cat([st_fea, refine_fea], dim=1)
        att_map = self.relu(self.conv10(att_map))
        att_map = self.sigmoid(self.conv11(att_map))

        return att_map


if __name__ == '__main__':
    pass
