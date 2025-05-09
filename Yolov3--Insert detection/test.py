if __name__ == "__main__":
  
    model = YOLOv3(num_classes=7)  #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt_path = "yolo_epoch9.pth"  # 要加载的文件名
    state_dict = torch.load(ckpt_path, map_location=device)  # CPU / GPU 通用
    model.load_state_dict(state_dict, strict=True)  # strict=True：键必须完全对上
    model.float().to(device).eval()

    total_results = []
    CONF_TH = 0.01
    IOU_TH  = 0.45
    test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = torch.tensor(img_data).to(device)
        img_scale = torch.tensor(img_scale_data).to(device)
        print(img.shape)
        outputs = model.forward(img)
        print(len(outputs))
        print(outputs[2].shape)
        boxes, scores = decode_yolo_multiscale(
            outputs,
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
            num_classes=7)
    
        print(boxes.shape, scores.shape)
        conf_thres = 0.1
        # 1. 找出最高类别分数
        obj_cls_scores, cls_idx = scores[..., 1:].max(dim=-1)  # [B,N]
    
        mask = obj_cls_scores > conf_thres  # True / False
    
        # 2. 根据 mask 过滤
        boxes_filtered = boxes[mask]  # [M,4]
        labels_filtered = cls_idx[mask]  # [M]
        scores_filtered = obj_cls_scores[mask]  # [M]  ← 单值置信度
        print(boxes_filtered.shape,scores_filtered.shape)
        from torchvision.ops import batched_nms,nms
    
        iou_thr = 0.6
        keep_idx = batched_nms(
            boxes_filtered,  # Tensor
            scores_filtered,  # Tensor
            labels_filtered,  # Tensor   ← 关键：按类别独立 NMS
            iou_thr)
    
        #keep_idx = keep_idx[:200]  # 选前 200 个（可选）
        boxes_nms = boxes_filtered[keep_idx]
        scores_nms = scores_filtered[keep_idx]
        labels_nms = labels_filtered[keep_idx]
    
        keep_idx_2=nms(boxes_nms,scores_nms,0.45)
        boxes_nms = boxes_nms[keep_idx_2]
        scores_nms = scores_nms[keep_idx_2]
        labels_nms = labels_nms[keep_idx_2]
    
        class_names = ["bg", "class1", "class2", "class3",
                       "class4", "class5", "class6", "class7"]  # 前面 bg 只是占位
        print(img_name)
        draw_boxes_save(
            img_path=f"/mnt/disk1/LY/深度学习/inserts/insects/test/images/{img_name[0]}.jpeg",
            boxes=boxes_nms.cpu().detach().numpy(),
            scores=scores_nms.cpu().detach().numpy(),
            labels=labels_nms.cpu().detach().numpy(),
            class_names=class_names[1:],  # 如果不想管 bg
            score_thr=0.3,
            out_dir="./vis_results")
