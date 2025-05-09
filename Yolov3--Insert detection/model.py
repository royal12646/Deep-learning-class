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

def sigmoid(x):
    return torch.sigmoid(x)

@torch.no_grad()
def yolo_box_xxyy_torch(pred, anchors, num_classes, downsample):
    """
    将 YOLOv3 网络输出的特征图 (tx, ty, tw, th) 解码为
    归一化 (x1, y1, x2, y2) 边界框坐标。

    Args
    ----
    pred : Tensor[ B, C, H, W ]
        - 网络的 raw 输出特征图
        - C = num_anchors * (5 + num_classes)
    anchors : list or Tensor
        - 长度 = 2 × num_anchors，格式 [w1, h1, w2, h2, ...]
        - 单位与输入尺寸相同（一般为像素）
    num_classes : int
        - 类别数
    downsample : int
        - 该特征图相对原图的 stride
          *P0=32, P1=16, P2=8* 等

    Returns
    -------
    boxes_xyxy : Tensor[ B, H, W, num_anchors, 4 ]
        - 4 为 (x1, y1, x2, y2)
        - 坐标已归一化到 0~1
    """
    # -------------------------------------------------------------
    # 1. 基本维度信息
    # -------------------------------------------------------------
    B, C, H, W = pred.shape                      # batch, channels, height, width
    num_anchors = len(anchors) // 2
    assert C == num_anchors * (num_classes + 5), \
        "channel size {} 与设定不符".format(C)

    # -------------------------------------------------------------
    # 2. 重新排列维度： [B, NA, 5+CLS, H, W] → [B, H, W, NA, 4]
    # -------------------------------------------------------------
    pred = pred.view(B, num_anchors, num_classes + 5, H, W)   # 先 group by anchor
    pred_loc = pred[:, :, 0:4, :, :]                          # 取 tx,ty,tw,th
    pred_loc = pred_loc.permute(0, 3, 4, 1, 2)                # B H W NA 4

    # -------------------------------------------------------------
    # 3. 网格坐标 (cx, cy) 以及 anchor 尺寸 (pw, ph)
    # -------------------------------------------------------------
    device = pred.device
    # 生成网格 y,x 索引，shape = [H, W]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij")
    grid_xy = torch.stack((grid_x, grid_y), dim=-1)           # H W 2
    grid_xy = grid_xy.unsqueeze(0).unsqueeze(3)               # 1 H W 1 2 (便于广播)

    # anchors → Tensor[NA,2] → reshape 为 1×1×1×NA×2
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchor_wh = anchors.view(num_anchors, 2)                  # NA 2
    anchor_wh = anchor_wh.view(1, 1, 1, num_anchors, 2)       # 1 H W NA 2

    # -------------------------------------------------------------
    # 4. 解码公式
    #    bx = (sigmoid(tx) + cx) / W
    #    by = (sigmoid(ty) + cy) / H
    #    bw = exp(tw) * pw / input_w
    #    bh = exp(th) * ph / input_h
    # -------------------------------------------------------------
    input_w = W * downsample
    input_h = H * downsample

    # ① 中心点坐标归一化
    box_xy = (torch.sigmoid(pred_loc[..., 0:2]) + grid_xy)   # B H W NA 2
    box_xy = box_xy / torch.tensor([W, H], device=device)

    # ② 宽高归一化
    box_wh = torch.exp(pred_loc[..., 2:4]) * anchor_wh        # B H W NA 2
    box_wh = box_wh / torch.tensor([input_w, input_h], device=device)

    # -------------------------------------------------------------
    # 5. 转成 (x1,y1,x2,y2)
    # -------------------------------------------------------------
    x1y1 = box_xy - box_wh / 2.0
    x2y2 = box_xy + box_wh / 2.0
    boxes = torch.cat([x1y1, x2y2], dim=-1)                   # B H W NA 4
    boxes.clamp_(0.0, 1.0)                                    # 保证 0~1

    return boxes

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
