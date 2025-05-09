import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNLayer(nn.Module):
    """Conv2d + BatchNorm2d + LeakyReLU"""
    def __init__(self, ch_in, ch_out, k=3, s=1, p=0,
                 groups=1, act="leaky"):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, p,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(ch_out)
        self.act  = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == "leaky":
            x = F.leaky_relu(x, 0.1, inplace=True)
        return x


class DownSample(nn.Module):
    """下采样：stride=2 的 3×3 卷积"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = ConvBNLayer(ch_in, ch_out,
                                k=3, s=2, p=1)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    """残差单元：1×1 + 3×3，然后与输入相加"""
    def __init__(self, ch_in, ch_mid):
        super().__init__()
        self.conv1 = ConvBNLayer(ch_in,   ch_mid,   k=1, s=1, p=0)
        self.conv2 = ConvBNLayer(ch_mid,  ch_mid*2, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class LayerWarp(nn.Module):
    """若干 BasicBlock 堆叠成一个 stage"""
    def __init__(self, ch_in, ch_mid, num_block):
        super().__init__()
        layers = [BasicBlock(ch_in, ch_mid)]
        layers += [BasicBlock(ch_mid*2, ch_mid)
                   for _ in range(1, num_block)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


# --------------------- DarkNet‑53 Backbone ---------------------
DarkNet_cfg = {53: [1, 2, 8, 8, 4]}          # 每个 stage 的 block 数
class DarkNet53(nn.Module):
    """
    返回三个尺度特征：
        C0: 1024 × 20 × 20   (stride 32)
        C1:  512 × 40 × 40   (stride 16)
        C2:  256 × 80 × 80   (stride 8)
    """
    def __init__(self):
        super().__init__()

        # Stem
        self.conv0      = ConvBNLayer(3, 32, k=3, s=1, p=1)
        self.downsample0 = DownSample(32, 64)

        stages_cfg = DarkNet_cfg[53]
        in_chs     = [ 64, 128, 256, 512, 1024]
        mid_chs    = [ 32,  64, 128, 256,  512]

        # 生成 5 个 LayerWarp
        self.stages = nn.ModuleList([
            LayerWarp(in_chs[i], mid_chs[i], stages_cfg[i])
            for i in range(len(stages_cfg))
        ])

        # 4 个 DownSample 用于各 stage 之间
        self.downsamples = nn.ModuleList([
            DownSample(in_chs[i], in_chs[i+1])
            for i in range(len(stages_cfg)-1)
        ])

    def forward(self, x):
        x = self.conv0(x)
        x = self.downsample0(x)

        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)          # 残差堆叠
            outputs.append(x)     # 保存输出
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # 返回 C0, C1, C2（与 Paddle 顺序保持一致）
        return outputs[-1], outputs[-2], outputs[-3]


class YoloDetectionBlock(nn.Module):
    """
    YOLOv3 Detection Block
    输入通道：ch_in
    内部通道：ch_out
    输出：
        route ──> 给上一层 FPN 的 Upsample & Concat
        tip   ──> 接 Detect Head（预测分类 + 位置 + 置信度）
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        assert ch_out % 2 == 0, f"channel {ch_out} cannot be divided by 2"

        self.conv0 = ConvBNLayer(ch_in,   ch_out,     k=1, s=1, p=0)
        self.conv1 = ConvBNLayer(ch_out,  ch_out * 2, k=3, s=1, p=1)
        self.conv2 = ConvBNLayer(ch_out * 2, ch_out,  k=1, s=1, p=0)
        self.conv3 = ConvBNLayer(ch_out,  ch_out * 2, k=3, s=1, p=1)
        self.route = ConvBNLayer(ch_out * 2, ch_out,  k=1, s=1, p=0)
        self.tip   = ConvBNLayer(ch_out,  ch_out * 2, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        route = self.route(x)  # 给 FPN 上采样分支
        tip   = self.tip(route)  # 给 Detect Head
        return route, tip


class Upsample(torch.nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale, mode="nearest")

class YOLOv3(torch.nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.block = DarkNet53()                      # 返回 C0, C1, C2
        self.yolo_blocks = torch.nn.ModuleList()
        self.route_blocks_2 = torch.nn.ModuleList()
        self.block_outputs  = torch.nn.ModuleList()
        self.upsample = Upsample()

        for i in range(3):
            ch_in = 1024 // (2 ** i)                  # C0:1024, C1:512, C2:256
            ch_route = ch_in // 2                     # 512,256,128

            # YoloDetectionBlock
            detect = YoloDetectionBlock(
                ch_in=ch_in if i == 0 else ch_in + ch_route,
                ch_out=ch_route)
            self.yolo_blocks.append(detect)

            # 1×1 卷积输出 pi
            num_filters = 3 * (self.num_classes + 5)
            self.block_outputs.append(
                torch.nn.Conv2d(ch_route * 2, num_filters, 1))

            # route2 卷积用于上采样分支
            if i < 2:
                self.route_blocks_2.append(
                    ConvBNLayer(ch_in=ch_route,
                                ch_out=ch_route // 2,
                                k=1, s=1, p=0))

    def forward(self, x):
        outputs = []
        blocks = self.block(x)            # [C0,C1,C2]  深→浅
        route = None
        for i, c_i in enumerate(blocks):
            if i > 0:                     # concat r_{i-1} 与 c_i
                c_i = torch.cat([route, c_i], dim=1)

            route, tip = self.yolo_blocks[i](c_i)
            p_i = self.block_outputs[i](tip)          # pi
            outputs.append(p_i)

            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)          # 上采样

        return outputs   # [P0, P1, P2]
