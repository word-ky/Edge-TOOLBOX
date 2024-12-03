import torch
import torch.nn as nn
import torch.nn.functional as F

import math
class Attention(nn.Module):
    def __init__(self, in_planes, K):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Conv2d(in_planes, K, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        att = self.avgpool(x)

        att = self.net(att)

        att = att.view(x.shape[0], -1)

        return self.sigmoid(att)

class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=1, groups=1, K=4):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes, K=K)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups,
                                               kernel_size[0], kernel_size[1]), requires_grad=True)

    def forward(self, x):

        N, in_planels, H, W = x.shape
        softmax_att = self.attention(x)

        x = x.view(1, -1, H, W)



        weight = self.weight

        weight = weight.view(self.K, -1)


        aggregate_weight = torch.mm(softmax_att, weight)

        aggregate_weight = aggregate_weight.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size[0], self.kernel_size[1])


        output = F.conv2d(x, weight=aggregate_weight,
                          stride=self.stride, padding=self.padding,
                          groups=self.groups*N)

        output = output.view(N, self.out_planes, H, W)

        return output




class CondConv_down(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=1, groups=1, K=4):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_planes=in_planes, K=K)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups,
                                               kernel_size[0], kernel_size[1]), requires_grad=True)

    def forward(self, x):

        N, in_planels, H, W = x.shape
        softmax_att = self.attention(x)

        x = x.view(1, -1, H, W)


        weight = self.weight

        weight = weight.view(self.K, -1)


        aggregate_weight = torch.mm(softmax_att, weight)

        aggregate_weight = aggregate_weight.view(
            N*self.out_planes, self.in_planes//self.groups,
            self.kernel_size[0], self.kernel_size[1])


        output = F.conv2d(x, weight=aggregate_weight,
                          stride=self.stride, padding=self.padding,
                          groups=self.groups*N)

        output = output.view(N, self.out_planes, H//2, W//2)
        # print(output.shape)
        return output

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class q_Split1(nn.Module):
    def __init__(self, dim=1):
        super(q_Split1, self).__init__()

        self.dim = dim

    def forward(self, input):

        splits = torch.chunk(input, 2, dim=self.dim)
        return splits






class CropLayer(nn.Module):


    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]

        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):


        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]



class DSNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(DSNet, self).__init__()
        self.split = q_Split1()
        self.channel_shuff = ChannelShuffle(groups=8)
        self.deploy = deploy

        out_channels = out_channels // 4
        in_channels = in_channels // 2
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            center_offset_from_origin_border = padding - kernel_size // 2

            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.con_ver_cov = CondConv_down(in_planes=in_channels, out_planes=out_channels, kernel_size=[3, 1],
                                               stride=stride, padding=ver_conv_padding)
            self.con_hor_cov = CondConv_down(in_planes=in_channels, out_planes=out_channels, kernel_size=[1, 3],
                                               stride=stride, padding=hor_conv_padding)
            self.con_square_cov = CondConv_down(in_planes=in_channels, out_planes=out_channels, kernel_size=[3, 3],
                                               stride=stride)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.square_cov = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels*2, kernel_size=3,
                                      stride=stride, padding=1
                                      )

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

    # forward function
    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            split = self.split(input)
            input = self.square_cov(input)
            # print(input.shape)
            # print(split[0].shape)
            # square_outputs = self.con_square_cov(split[0])
            # square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())

            vertical_outputs = self.ver_conv_crop_layer(split[0])
            vertical_outputs = self.con_ver_cov(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.shape)

            horizontal_outputs = self.hor_conv_crop_layer(split[1])
            horizontal_outputs = self.con_hor_cov(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)

            # print(horizontal_outputs.shape)
            out = torch.concat((input, vertical_outputs, horizontal_outputs), dim=1)
            # print(out.shape)
            out = self.channel_shuff(out)
            return out


class q_Split(nn.Module):
    def __init__(self, dim=1):
        super(q_Split, self).__init__()

        self.dim = dim

    def forward(self, input):

        splits = torch.chunk(input, 4, dim=self.dim)
        return splits






class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]

        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):


        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]



class BasicNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(BasicNet, self).__init__()
        self.split = q_Split()
        self.channel_shuff = ChannelShuffle(groups=8)
        self.deploy = deploy

        out_channels = out_channels//4
        in_channels = in_channels // 4
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            center_offset_from_origin_border = padding - kernel_size // 2

            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.con_ver_cov = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=[3, 1],
                                               stride=stride, padding=ver_conv_padding)
            self.con_hor_cov = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=[1, 3],
                                               stride=stride, padding=hor_conv_padding)
            self.con_square_cov = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=[3, 3],
                                               stride=stride)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

    # forward function
    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            split = self.split(input)
            square_outputs = self.con_square_cov(split[1])
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())

            vertical_outputs = self.ver_conv_crop_layer(split[2])
            vertical_outputs = self.con_ver_cov(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())

            horizontal_outputs = self.hor_conv_crop_layer(split[3])
            horizontal_outputs = self.con_hor_cov(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())

            out = torch.concat((split[0], square_outputs, vertical_outputs, horizontal_outputs), dim=1)
            # print(out.shape)
            out = self.channel_shuff(out)
            return out

class SimAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambd=1e-4
    def forward(self, X):
        n = X.shape[2] * X.shape[3] - 1
        print(X.shape)
        print(X.mean(dim=[2, 3]).shape)
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + self.lambd)) + 0.5
        return X * torch.sigmoid(E_inv)


def autopad(k, p=None, d=1):

    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p



class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):

    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.stem = Conv(3, 64, s=2)
        self.DSnet1 = DSNet(in_channels=64, out_channels=128, kernel_size=3)
        self.DSnet2 = DSNet(in_channels=128, out_channels=256, kernel_size=3)
        self.DSnet3 = DSNet(in_channels=256, out_channels=512, kernel_size=3)
        self.basic1 = BasicNet(in_channels=128, out_channels=128, kernel_size=3)
        self.basic2 = BasicNet(in_channels=256, out_channels=256, kernel_size=3)
        self.basic3 = BasicNet(in_channels=512, out_channels=512, kernel_size=3)
        self.Sim = SimAM()

    def forward(self, input):
        input = self.stem(input)
        # print(input.shape)
        input = self.DSnet1(input)
        # print(input.shape)
        input = self.basic1(input)
        feat1 = input
        input = self.DSnet2(input)
        input = self.Sim(input)
        input = self.basic2(input)
        feat2 = input
        input = self.DSnet3(input)
        input = self.Sim(input)
        input = self.basic3(input)
        feat3 = input
        print(feat1.shape, feat2.shape, feat3.shape)
        return feat1, feat2, feat3

import torch
import torchvision

from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = backbone().to(device)
