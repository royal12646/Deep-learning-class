import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import random
import time
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


import matplotlib.patches as patches

############################################################################数据处理

def multi_box_iou_xywh(box1, box2):
    """
    In this case, box1 or box2 can contain multi boxes.
    Only two cases can be processed in this method:
       1, box1 and box2 have the same shape, box1.shape == box2.shape
       2, either box1 or box2 contains only one box, len(box1) == 1 or len(box2) == 1
    If the shape of box1 and box2 does not match, and both of them contain multi boxes, it will be wrong.
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."


    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (
        boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (
        boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(
        axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (
        boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (
        boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()




def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2


def get_img_data_from_file(record):
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
    """
    im_file = record['im_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), \
        "image height of {} inconsistent in record({}) and img file({})".format(
            im_file, h, img.shape[0])

    assert img.shape[1] == int(w), \
        "image width of {} inconsistent in record({}) and img file({})".format(
            im_file, w, img.shape[1])

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)


def get_insect_names():
    """
    return a dict, as following,
        {'Boerner': 0,
         'Leconte': 1,
         'Linnaeus': 2,
         'acuminatus': 3,
         'armandi': 4,
         'coleoptera': 5,
         'linnaeus': 6
        }
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


def get_annotations(cname2cid, datadir):
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), ), dtype=np.int32)
        is_crowd = np.zeros((len(objs), ), dtype=np.int32)
        difficult = np.zeros((len(objs), ), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            # 这里使用xywh格式来表示目标物体真实框
            gt_bbox[i] = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img



def visualize(srcimg, img_enhance, save_path, gtbox=None):
    plt.figure(num=2, figsize=(8, 4), facecolor='black')  # 整个背景黑色
    plt.subplot(1, 2, 1)
    plt.title('Src Image', color='white')
    plt.axis('off')
    plt.imshow(srcimg)

    plt.subplot(1, 2, 2)
    plt.title('Enhance Image', color='white')
    plt.axis('off')
    img_show = np.clip(img_enhance, 0, 255).astype(np.uint8)

    plt.imshow(img_show)

    plt.tight_layout()
    plt.savefig(save_path, facecolor='black')  # 保存时也保留黑底
    plt.close()

# 随机填充
def random_expand(img,
                  gtboxes,
                  max_ratio=1.6,
                  fill=None,
                  keep_ratio=False,
                  thresh=0.5):
    r=random.random()
    if r> thresh:
        return img, gtboxes

    if max_ratio < 1.0:
        print(1)
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes

# 随机裁剪
def random_crop(img,
                boxes,
                labels,
                scales=[0.3, 1.0],
                max_ratio=2.0,
                constraints=None,
                max_trial=50):
    if len(boxes) == 0:
        return img, boxes

    if not constraints:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0),
                       (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w,
                                  (crop_y + crop_h / 2.0) / h,
                                  crop_w / float(w), crop_h / float(h)]])

            iou = multi_box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels

# 随机缩放
def random_interp(img, size, interp=cv2.INTER_AREA):
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img


# 随机翻转
def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes

# 图像增广方法汇总
def image_augment(img, gtboxes, gtlabels, size, means=None):
    # 随机改变亮暗、对比度和颜色等
    img = random_distort(img)
    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    # 随机裁剪
    img, gtboxes, gtlabels, = random_crop(img, gtboxes, gtlabels)
    # 随机缩放
    img = random_interp(img, size)
    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)
    # 随机打乱真实框排列顺序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')

# 随机打乱真实框排列顺序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]


def get_img_data(record, size=640):
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img, gt_boxes, gt_labels, scales


# ---------------------  数据集 ---------------------
class TrainDataset(Dataset):
    """返回
       img  : Tensor  [3, 640, 640]  (float32)
       boxes: Tensor  [50, 4]        (float32)
       labels:Tensor  [50]           (int64)
       im_shape: Tensor [2]          (int32)  (H,W)
    """
    def __init__(self, datadir, mode):
        self.datadir = datadir
        cname2cid    = get_insect_names()
        self.records = get_annotations(cname2cid, datadir)
        self.img_size = 640   # 固定 640 × 640
        self.mode = mode

        print(self.mode)
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=self.img_size)


        # numpy → torch，并保持形状/数据类型
        img       = torch.from_numpy(img)                         # [3,640,640] float32
        gt_bbox   = torch.from_numpy(gt_bbox)                     # [50,4]
        gt_labels = torch.from_numpy(gt_labels).long()            # int64
        im_shape  = torch.as_tensor(im_shape, dtype=torch.int32)  # [2]

        return img, gt_bbox, gt_labels, im_shape


# ---------------------  DataLoader ---------------------
def collate_fn(batch):
    """
    将 list[(img, boxes, labels, im_shape), ...] 组合成批
    数据集中 boxes 已经 pad 到固定 50，因此可以直接 stack。
    """
    imgs, boxes, labels, shapes = zip(*batch)
    return (torch.stack(imgs,   dim=0),    # [B,3,640,640]
            torch.stack(boxes,  dim=0),    # [B,50,4]
            torch.stack(labels, dim=0),    # [B,50]
            torch.stack(shapes, dim=0))    # [B,2]

# 将 list形式的batch数据 转化成多个array构成的tuple
def make_test_array(batch_data):
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype = 'float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array

# 测试数据读取
def test_data_loader(datadir, batch_size= 10, test_image_size=608, mode='test'):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    image_names = os.listdir(datadir)
    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in image_names:
            file_path = os.path.join(datadir, image_name)
            print(file_path)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))
            img = out_img #np.transpose(out_img, (2,0,1))
            im_shape = [H, W]

            batch_data.append((image_name.split('.')[0], img, im_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader

##################################################################################

@torch.jit.script     # 可选：JIT 加速
def box_iou_xywh(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # 分离并计算两框的角点
    x1min = box1[..., 0] - box1[..., 2] * 0.5
    y1min = box1[..., 1] - box1[..., 3] * 0.5
    x1max = box1[..., 0] + box1[..., 2] * 0.5
    y1max = box1[..., 1] + box1[..., 3] * 0.5
    area1 = box1[..., 2] * box1[..., 3]

    x2min = box2[..., 0] - box2[..., 2] * 0.5
    y2min = box2[..., 1] - box2[..., 3] * 0.5
    x2max = box2[..., 0] + box2[..., 2] * 0.5
    y2max = box2[..., 1] + box2[..., 3] * 0.5
    area2 = box2[..., 2] * box2[..., 3]

    # 交集宽高
    inter_w = torch.clamp_min(torch.minimum(x1max, x2max) - torch.maximum(x1min, x2min), 0.)
    inter_h = torch.clamp_min(torch.minimum(y1max, y2max) - torch.maximum(y1min, y2min), 0.)
    inter   = inter_w * inter_h

    union = area1 + area2 - inter
    iou   = inter / (union + 1e-10)          # 避免除零
    return iou


@torch.no_grad()
def get_objectness_label(img,
                               gt_boxes,          # [B, 50, 4]  (xywh, 0‑1)
                               gt_labels,         # [B, 50]
                               iou_threshold=0.7,
                               anchors=(116,90, 156,198, 373,326),
                               num_classes=7,
                               downsample=32):
    """
    返回：
        label_objectness      [B, NA, H, W]
        label_location        [B, NA, 4, H, W]
        label_classification  [B, NA, C, H, W]
        scale_location        [B, NA, H, W]
    全部 float32，device 与 img 相同
    """
    device   = img.device
    B, _, H_in, W_in = img.shape

    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    NA      = anchors.numel() // 2
    anchor_wh = anchors.view(NA, 2)                            # [NA,2]

    H_feat = H_in // downsample
    W_feat = W_in // downsample

    # 初始化标签
    label_objectness     = torch.zeros((B, NA, H_feat, W_feat), device=device)
    label_classification = torch.zeros((B, NA, num_classes, H_feat, W_feat), device=device)
    label_location       = torch.zeros((B, NA, 4, H_feat, W_feat), device=device)
    scale_location       = torch.ones ((B, NA, H_feat, W_feat), device=device)

    for n in range(B):
        # 取出有效 GT（w,h>0）
        boxes_n   = gt_boxes[n]                                # [50,4]
        labels_n  = gt_labels[n]                               # [50]
        mask_valid = (boxes_n[:, 2] > 1e-3) & (boxes_n[:, 3] > 1e-3)
        boxes_n   = boxes_n[mask_valid]
        labels_n  = labels_n[mask_valid]

        if boxes_n.numel() == 0:
            continue

        for gt, cls in zip(boxes_n, labels_n):
            cx, cy, gw, gh = gt                               # all 0‑1
            i_grid = int(cy * H_feat)
            j_grid = int(cx * W_feat)

            # -------------- 与各 anchor IoU ----------------
            gt_tmp   = torch.tensor([0.,0., gw, gh], device=device)
            anchor_tmp = torch.cat([torch.zeros((NA,2), device=device),  # zeros x,y
                                    anchor_wh / torch.tensor([W_in, H_in], device=device)], dim=1)  # 归一化

            ious = box_iou_xywh(gt_tmp, anchor_tmp)           # [NA]
            k = torch.argmax(ious)                            # best anchor

            # -------------- 写入标签 -----------------------
            label_objectness[n, k, i_grid, j_grid] = 1.0
            label_classification[n, k, int(cls), i_grid, j_grid] = 1.0

            dx = cx * W_feat - j_grid
            dy = cy * H_feat - i_grid
            dw = torch.log(gw * W_in / anchor_wh[k, 0] + 1e-16)
            dh = torch.log(gh * H_in / anchor_wh[k, 1] + 1e-16)

            label_location[n, k, 0, i_grid, j_grid] = dx
            label_location[n, k, 1, i_grid, j_grid] = dy
            label_location[n, k, 2, i_grid, j_grid] = dw
            label_location[n, k, 3, i_grid, j_grid] = dh

            scale_location[n, k, i_grid, j_grid] = 2.0 - gw * gh

    return (label_objectness.float(),
            label_location.float(),
            label_classification.float(),
            scale_location.float())

# --------------------- 基础模块 ---------------------
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

# 挑选出跟真实框IoU大于阈值的预测框
@torch.no_grad()
def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold=0.5):
    """
    Args
    ----
    pred_box : Tensor [B, H, W, A, 4]   (x1,y1,x2,y2 归一化坐标)
    gt_boxes : Tensor [B, M, 4]         (同样坐标；若不足 M，用 0 填充)
    iou_threshold : float
    Returns
    -------
    ret_inds : BoolTensor [B, A, H, W]  (与 Paddle 版一致，已转置)
    """

    B, H, W, A, _ = pred_box.shape
    M = gt_boxes.shape[1]                                # 每图最多 M 个 GT

    # -------------- 预处理：获得 valid GT mask --------------
    # 宽或高 < 1e‑3 视为无效
    gt_wh   = gt_boxes[..., 2:]                          # w,h
    valid_gt = (gt_wh[..., 0] > 1e-3) & (gt_wh[..., 1] > 1e-3)   # [B,M]

    # -------------- 扩维做广播 --------------
    #   pred_box: B H W A 1 4
    #   gt_box : B 1 1 1 M 4
    pb = pred_box.unsqueeze(-2)                          # → B H W A 1 4
    gb = gt_boxes.unsqueeze(1).unsqueeze(1).unsqueeze(1) # → B 1 1 1 M 4

    # 分离坐标
    pb_x1, pb_y1, pb_x2, pb_y2 = pb[..., 0], pb[..., 1], pb[..., 2], pb[..., 3]
    gb_x1 = gb[..., 0] - gb[..., 2] / 2.0
    gb_y1 = gb[..., 1] - gb[..., 3] / 2.0
    gb_x2 = gb[..., 0] + gb[..., 2] / 2.0
    gb_y2 = gb[..., 1] + gb[..., 3] / 2.0

    # -------------- IoU --------------
    inter_x1 = torch.maximum(pb_x1, gb_x1)
    inter_y1 = torch.maximum(pb_y1, gb_y1)
    inter_x2 = torch.minimum(pb_x2, gb_x2)
    inter_y2 = torch.minimum(pb_y2, gb_y2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_pb = (pb_x2 - pb_x1) * (pb_y2 - pb_y1)
    area_gb = (gb_y2 - gb_y1) * (gb_x2 - gb_x1)

    union = area_pb + area_gb - inter
    iou   = inter / torch.clamp(union, min=1e-10)         # B H W A M

    # -------------- 过滤无效 GT --------------
    valid_mask = valid_gt.view(B, 1, 1, 1, M)             # broadcast
    iou = iou * valid_mask.float()

    # -------------- 满足阈值的索引 --------------
    above_thresh = (iou > iou_threshold)                  # Bool B H W A M
    # 若一个预测框对任意 GT 满足阈值，则该框计为正样本
    above_any = above_thresh.any(dim=-1)                  # B H W A

    # 转置为 [B, A, H, W] 与 Paddle 版一致
    ret_inds = above_any.permute(0, 3, 1, 2).contiguous()

    return ret_inds

def label_objectness_ignore(label_objectness, iou_above_thresh_indices):
    # 注意：这里不能简单的使用 label_objectness[iou_above_thresh_indices] = -1，
    #         这样可能会造成label_objectness为1的点被设置为-1了
    #         只有将那些被标注为0，且与真实框IoU超过阈值的预测框才被标注为-1
    negative_indices = (label_objectness < 0.5)
    ignore_indices = negative_indices * iou_above_thresh_indices
    label_objectness[ignore_indices] = -1
    return label_objectness


def get_loss(output,
             label_objectness,
             label_location,
             label_classification,
             scales,
             num_anchors=3,
             num_classes=7):
    """
    Args
        output                : Tensor [B, C, H, W]   (模型原始输出)
        label_objectness      : [B, NA, H, W]
        label_location        : [B, NA, 4, H, W]      (tx,ty,tw,th)
        label_classification  : [B, NA, num_classes, H, W]
        scales                : [B, NA, H, W]         (IoU‑based 位置权重)
    Returns
        total_loss : scalar Tensor
    """

    B, C, H, W = output.shape
    # 1) reshape → [B, NA, 5+cls, H, W]
    out = output.view(B, num_anchors, num_classes + 5, H, W)

    # ---------- 目标置信度损失 ----------
    pred_obj = out[:, :, 4, :, :]                               # [B,NA,H,W]
    ignore_mask = (label_objectness < 0)       # True at -1
    tgt_obj = label_objectness.clone()
    tgt_obj[ignore_mask] = 0                   # -1 → 0   (占位)

    loss_obj = F.binary_cross_entropy_with_logits(
        pred_obj, tgt_obj, reduction="none")
    loss_obj[ignore_mask] = 0                  # 忽略样本不计入损失

    pos_mask = (label_objectness > 0).float()  # 正样本掩码
    pos_mask.requires_grad_(False)



    # ---------- 位置损失 ----------
    tx, ty, tw, th = out[:, :, 0, :, :], out[:, :, 1, :, :], \
                     out[:, :, 2, :, :], out[:, :, 3, :, :]

    dx, dy = label_location[:, :, 0, :, :], label_location[:, :, 1, :, :]
    tw_lbl, th_lbl = label_location[:, :, 2, :, :], label_location[:, :, 3, :, :]

    loss_x = F.binary_cross_entropy_with_logits(tx, dx, reduction="none")
    loss_y = F.binary_cross_entropy_with_logits(ty, dy, reduction="none")
    loss_w = torch.abs(tw - tw_lbl)
    loss_h = torch.abs(th - th_lbl)
    loss_loc = (loss_x + loss_y + loss_w + loss_h)              # [B,NA,H,W]
    loss_loc = loss_loc * scales                                # 加权
    loss_loc = loss_loc * pos_mask                              # 只算正样本

    # ---------- 分类损失 ----------
    pred_cls = out[:, :, 5:5 + num_classes, :, :]               # [B,NA,C,H,W]
    loss_cls = F.binary_cross_entropy_with_logits(
        pred_cls, label_classification, reduction="none")

    loss_cls = loss_cls.sum(dim=2)                              # 按类别求和 → [B,NA,H,W]
    loss_cls = loss_cls * pos_mask

    # ---------- 汇总 ----------
    total = loss_obj + loss_loc + loss_cls                      # [B,NA,H,W]
    total = total.sum(dim=(1, 2, 3))                            # 每张图求和
    total = total.mean()                                        # batch 平均

    return total

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

def _flatten_selected_anchors(all_anchors, mask):
    """从 9 个锚框里挑选指定 index"""
    sel = []
    for idx in mask:
        sel.extend(all_anchors[idx*2: idx*2+2])
    return sel            # list 长度 = len(mask)*2


@torch.no_grad()
def build_targets(out, img,gt_boxes, gt_labels,
                  anchors_layer, ignore_thresh,
                  num_classes, downsample):
    """
    根据某一层输出生成：
      label_objectness, label_location, label_classification, scale_location
    并对 IoU > ignore_thresh 的负样本做 ignore 处理
    """
    (l_obj, l_loc, l_cls, s_loc) = get_objectness_label(
        img=img,  # 只要形状
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        anchors=anchors_layer,
        num_classes=num_classes,
        downsample=downsample)

    # IoU > thresh 的负样本忽略
    pred_boxes = yolo_box_xxyy_torch(out, anchors_layer,
                                     num_classes=num_classes,
                                     downsample=downsample)
    ignore_mask = get_iou_above_thresh_inds(
        pred_boxes, gt_boxes, iou_threshold=ignore_thresh)     # [B,NA,H,W]
    l_obj = label_objectness_ignore(l_obj, ignore_mask)        # 置 -1

    # Stop‑grad（在 PyTorch 中只需保证 target 不参与梯度即可）
    return l_obj, l_loc, l_cls, s_loc

def get_loss_multiscale(outputs,         # [P0, P1, P2]
                        gt_boxes,        # [B,50,4]
                        gt_labels,       # [B,50]
                        *,
                        anchors,         # len=18
                        anchor_masks,    # [[6,7,8],[3,4,5],[0,1,2]]
                        ignore_thresh=0.7,
                        num_classes=7,
                        device=None):
    """
    返回三尺度总损失（标量 Tensor）
    """
    if device is None:
        device = outputs[0].device

    total_loss = 0.0
    downsample = 32                       # P0 -> 32, P1 ->16, P2 -> 8

    for i, out in enumerate(outputs):     # 循环 P0,P1,P2
        mask = anchor_masks[i]
        anchors_layer = _flatten_selected_anchors(anchors, mask)
        num_anchors_layer = len(mask)

        # ---------- 生成各类标签 ----------
        l_obj, l_loc, l_cls, s_loc = build_targets(
            out, img,gt_boxes, gt_labels,
            anchors_layer, ignore_thresh,
            num_classes, downsample)

        # ---------- 计算单层损失 ----------
        loss_i = get_loss(out,
                          l_obj.to(device),
                          l_loc.to(device),
                          l_cls.to(device),
                          s_loc.to(device),
                          num_anchors=num_anchors_layer,
                          num_classes=num_classes)

        total_loss = total_loss + loss_i
        downsample //= 2                  # 下一层 stride 减半

    return total_loss

def get_lr(optimizer,
                     base_lr: float = 3e-4,
                     lr_decay: float = 0.1):
    # 先把 optimizer 的初始 lr 设为 base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,  # 每 3 个 epoch 衰减一次
        gamma=0.4
    )
    return scheduler



def decode_yolo_feat(feat,            # Tensor [B, C, H, W]
                     anchors,         # list  len = 2*NA
                     num_classes: int,
                     stride: int):
    """
    返回:
        boxes_xyxy    [B, H, W, NA, 4]  (0‑1 归一化, xyxy)
        obj_prob      [B, NA, H, W]     (sigmoid 概率)
        cls_prob      [B, NA, num_classes, H, W] (sigmoid 概率)
    """
    B, C, H, W = feat.shape
    NA = len(anchors) // 2
    assert C == NA * (5 + num_classes), "channel mismatch!"

    # -------- 1. 解析输出 ----------
    feat = feat.view(B, NA, 5 + num_classes, H, W)

    # 先取出，再 permute 到 B,H,W,NA
    tx = feat[:, :, 0].permute(0, 2, 3, 1)  # [B,H,W,NA]
    ty = feat[:, :, 1].permute(0, 2, 3, 1)
    tw = feat[:, :, 2].permute(0, 2, 3, 1)
    th = feat[:, :, 3].permute(0, 2, 3, 1)

    obj_logit = feat[:, :, 4].permute(0, 2, 3, 1)  # [B,H,W,NA]
    cls_logit = feat[:, :, 5:].permute(0, 2, 3, 1, 4)  # [B,H,W,NA,C]

    # -------- 2. 生成网格 ----------
    device = feat.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij")
    grid_xy = torch.stack((grid_x, grid_y), dim=-1)      # H,W,2
    grid_xy = grid_xy.unsqueeze(0).unsqueeze(3)          # 1,H,W,1,2

    # -------- 3. anchor 尺寸 ----------
    anchor_wh = torch.tensor(anchors, device=device).view(NA, 2)  # NA,2
    anchor_wh = anchor_wh.view(1,1,1,NA,2)                       # 1,H,W,NA,2

    # -------- 4. 解码 ----------
    pred_xy = (torch.sigmoid(torch.stack((tx, ty), dim=-1)) + grid_xy) * stride
    pred_wh = torch.exp(torch.stack((tw, th), dim=-1)) * anchor_wh
    inp_size = stride * H
    boxes_xyxy = torch.cat((pred_xy - pred_wh/2,
                            pred_xy + pred_wh/2), dim=-1) / inp_size
    boxes_xyxy.clamp_(0., 1.)

    # -------- 5. 概率 ----------
    obj_prob = torch.sigmoid(obj_logit)
    cls_prob = torch.sigmoid(cls_logit)

    return boxes_xyxy, obj_prob, cls_prob

def _select_anchors(all_anchors, mask):
    """根据 mask 挑选 anchors 并扁平化"""
    sel = []
    for idx in mask:
        sel.extend(all_anchors[idx*2: idx*2+2])
    return sel                              # [w0,h0, w1,h1, w2,h2]

def decode_yolo_multiscale(outputs,        # [P0,P1,P2]  raw logits
                           anchor_masks,    # [[6,7,8],[3,4,5],[0,1,2]]
                           anchors,         # 全 9 枚 anchor
                           num_classes):
    """
    返回三尺度拼接后的：
        boxes_all [B, N, 4]   (xyxy, 0‑1)
        scores_all[B, N, C+1] (obj * cls，已含背景)
    """
    all_boxes   = []
    all_scores  = []

    strides = [32, 16, 8]                   # P0,P1,P2
    for feat, mask, stride in zip(outputs, anchor_masks, strides):
        anchors_layer = _select_anchors(anchors, mask)
        boxes, obj_p, cls_p = decode_yolo_feat(
            feat, anchors_layer, num_classes, stride)    # shapes 见上

        B, H, W, NA, _ = boxes.shape
        # 展平成 [B, H*W*NA, ...]
        boxes = boxes.reshape(B, -1, 4)
        obj_p = obj_p.reshape(B, -1, 1)                  # [B,N,1]
        cls_p = cls_p.permute(0,2,1,3,4).reshape(B, num_classes, -1) \
                     .permute(0,2,1)                     # [B,N,C]

        scores = obj_p * cls_p                           # 置信度 × 分类
        bg_score = 1.0 - obj_p                           # 背景
        scores = torch.cat([bg_score, scores], dim=-1)   # [B,N,C+1]

        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes_all  = torch.cat(all_boxes,  dim=1)            # [B, sumN, 4]
    scores_all = torch.cat(all_scores, dim=1)            # [B, sumN, C+1]
    return boxes_all, scores_all


# ---------- 配色工具：给每个类别分一条颜色 ----------
def _random_color(index):
    np.random.seed(index + 12345)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))

def draw_boxes_save(img_path,
                    boxes,           # Tensor/ndarray [N,4]  xyxy 0~1
                    scores,          # Tensor/ndarray [N]    0~1
                    labels,          # Tensor/ndarray [N]    int
                    class_names,     # list[str]  len = num_classes
                    score_thr=0.25,
                    out_dir="./vis"):
    """
    把检测结果画到原图并保存.

    Args
    ----
    img_path   : 原图文件路径
    boxes      : 归一化 xyxy
    scores     : 置信度
    labels     : 类别索引
    score_thr  : 置信度阈值(再次过滤,可选)
    out_dir    : 保存目录
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    H, W = img.shape[:2]

    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, labels):
        scores = np.squeeze(scores)  # (N,1) → (N,)
        if score < score_thr:
            continue
        # 像素坐标
        p1 = int(x1 * W), int(y1 * H)
        p2 = int(x2 * W), int(y2 * H)

        color = _random_color(int(cls))
        cv2.rectangle(img, p1, p2, color, thickness=2)

        text = f"{class_names[int(cls)]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text,
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=0.5, thickness=1)
        cv2.rectangle(img, p1, (p1[0] + tw, p1[1] - th - 4),
                      color, -1)                          # 文本背景
        cv2.putText(img, text, (p1[0], p1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # 保存
        save_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"[INFO] saved: {save_path}")
