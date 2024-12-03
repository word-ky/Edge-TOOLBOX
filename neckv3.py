import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
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

class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.oup, :, :]



class POOL(nn.Module):

    def __init__(self):
        super(POOL, self).__init__()
        self.pool_h = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = torch.nn.AdaptiveAvgPool2d((1, None))

        self.pool_h1 = torch.nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w1 = torch.nn.AdaptiveMaxPool2d((1, None))

    def forward(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(3, 2)   # [0,1,3,2]
        x_h1 = self.pool_h1(x)
        x_w1 = self.pool_w1(x).transpose(3, 2)
        y1 = torch.cat([x_h, x_w], dim=2)
        # print(y1.shape)
        y2 = torch.cat([x_h1, x_w1], dim=2)
        # print(y2.shape)
        y = torch.cat([y1, y2], dim=1)
        # print(y.shape)
        return y


class attention(nn.Module):

    def __init__(self, size):
        super(attention, self).__init__()
        self.POOL = POOL()
        self.size = size

    def forward(self, in_1, in_2):

        in_1 = self.POOL(in_1)
        in_2 = self.POOL(in_2)
        # print('in1')
        # print(in_1.shape)
        in_all = in_1.mul(in_2)
        in_all = in_all.mul(in_2)
        # print(in_all.shape)
        y = torch.split(in_all, self.size, dim=2)  # 按照行维度去分，每大块包含40个小块

        x_h = y[0]
        x_w = y[1]
        x_h = x_h.repeat(1, 1, 1, self.size)
        # print(x_h.shape)
        x_w = x_w.transpose(2, 3)
        x_w = x_w.repeat(1, 1, self.size, 1)
        # print(x_w.shape)

        # print(x_h.shape, x_w.shape)

        out = x_h + x_w
        return out


class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out

class BasicBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()

        self.channel = EfficientChannelAttention(planes)       # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        ECA_out = self.channel(x)
        out = x * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class concat(nn.Module):

    def __init__(self, size, channel):
        super(concat, self).__init__()
        self.channel = channel
        self.size = size

        self.SA = attention(size=self.size)
        self.CA = BasicBlock(in_planes=self.channel, planes=self.channel)
        self.cov = nn.Conv2d(in_channels=self.channel*2, out_channels=channel, kernel_size=1, stride=1)

    def forward(self, x_upsim, x_map):
        sa_out = self.SA(x_upsim, x_map)
        sa_out = self.cov(sa_out)
        ca_out = self.CA(x_upsim)
        out = sa_out + ca_out

        return out


class neck_up(nn.Module):

    def __init__(self):
        super(neck_up, self).__init__()
        self.sppf = SPPF(c1=512, c2=512)
        self.concat_u1 = concat(size=40, channel=512)
        self.concat_u2 = concat(size=80, channel=256)
        self.concat_u3 = concat(size=160, channel=128)
        self.ghost_u1 = GhostModule(inp=512, oup=256)
        self.ghost_u2 = GhostModule(inp=256, oup=128)
        self.ghost_u3 = GhostModule(inp=128, oup=128)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, map1, map2, map3, map4):
        map3_5 = self.sppf(map4)
        map3_up = self.upsample(map3_5)
        # print(map3_up.shape)
        map3_con = self.concat_u1(map3_up, map3)
        map3_g = self.ghost_u1(map3_con)    # 向后下向上数第一个
        map2_up = self.upsample(map3_g)

        map2_con = self.concat_u2(map2_up, map2)
        map2_g = self.ghost_u2(map2_con)   # 同上，第二个
        map1_up = self.upsample(map2_g)

        map1_con = self.concat_u3(map1_up, map1)
        map1_g = self.ghost_u3(map1_con)   # 同上，第三个

        # print(map1_g.shape)
        return map1_g, map2_g, map3_g

class neck_down(nn.Module):

    def __init__(self):
        super(neck_down, self).__init__()
        self.ghost_cbs1 = GhostModule(inp=128, oup=128, kernel_size=3, stride=2)
        self.ghost_cbs2 = GhostModule(inp=128, oup=256, kernel_size=3, stride=2)
        self.ghost_d1 = GhostModule(inp=128, oup=128)
        self.ghost_d2 = GhostModule(inp=256, oup=256)
        self.concat_d1 = concat(size=80, channel=128)
        self.concat_d2 = concat(size=40, channel=256)

    def forward(self, down1, down2, down3):
        down1_gb = self.ghost_cbs1(down1)   # down1第一个输出
        # print(down1_gb.shape)
        down1_con = self.concat_d1(down1_gb, down2)

        down1_g = self.ghost_d1(down1_con)  # 第二个输出
        down2_gb = self.ghost_cbs2(down1_g)
        down2_con = self.concat_d2(down2_gb, down3)

        down2_g = self.ghost_d2(down2_con)  # 第三个输出

        # print(down2_g.shape)
        return down1, down1_g, down2_g

class neck(nn.Module):

    def __init__(self):
        super(neck, self).__init__()
        self.up = neck_up()
        self.down = neck_down()

    def forward(self, map1, map2, map3, map4):
        maps1, maps2, maps3 = self.up(map1, map2, map3, map4)
        fea1, fea2, fea3 = self.down(maps1, maps2, maps3)
        # print(fea1.shape, fea2.shape, fea3.shape)
        return fea1, fea2, fea3


# import torch
# import torchvision
# # 导入torchsummary
# from torchsummary import summary
#
# # 需要使用device来指定网络在GPU还是CPU运行
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 建立神经网络模型，这里直接导入已有模型
# # model = model().to(device)
# model = neck().to(device)
# # 使用summary，注意输入维度的顺序
# summary(model, input_data=(torch.ones(1, 128, 160, 160), torch.ones(1, 256, 80, 80), torch.ones(1, 512, 40, 40),
#                            torch.ones(1, 512, 20, 20)))
