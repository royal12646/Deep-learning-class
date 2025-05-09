from torch.utils.tensorboard import SummaryWriter
import time, os
from model import YOLOv3
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time, os
from torch.utils.data import  DataLoader
from utils import get_insect_names,get_annotations
from utils import TrainDataset,build_subset_loader,test_data_loader
from utils import get_lr,get_loss_multiscale
from utils import evaluate_map


if __name__ == "__main__":
    INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus',
                    'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

    TRAINDIR = '/mnt/disk1/LY/深度学习/inserts/insects/train'
    TESTDIR = '/mnt/disk1/LY/深度学习/inserts/insects/test/images'
    VALIDDIR = '/mnt/disk1/LY/深度学习/inserts/insects/val'
    cname2cid = get_insect_names()
    records = get_annotations(cname2cid, TRAINDIR)

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TrainDataset(TRAINDIR, mode="train")
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')

    train_loader = DataLoader(train_dataset,
                              batch_size=10,
                              shuffle=True,
                              num_workers=2,
                              drop_last=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=10,
                              shuffle=False,
                              num_workers=2,
                              drop_last=False,
                              collate_fn=collate_fn)
    train_subset_loader = build_subset_loader(train_dataset,
                                              batch_size=10,
                                              n_samples=100,
                                              num_workers=2)

    test_loader=test_data_loader(TESTDIR, batch_size= 2, test_image_size=608, mode='test')
    model = YOLOv3(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4,
                                 weight_decay=5e-4)



    log_dir = "./runs/yolov3"            # 日志目录，可按实验命名
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    scheduler = get_lr(optimizer)
    MAX_EPOCH = 15
    global_step = 0
    for epoch in range(MAX_EPOCH):
        model.train()
        for i, (img, gt_boxes, gt_labels, _) in enumerate(train_loader):
            img, gt_boxes, gt_labels = img.to(device), gt_boxes.to(device), gt_labels.to(device)
            outputs = model(img)  #
            loss = get_loss_multiscale(outputs, gt_boxes, gt_labels,
                                       anchors=ANCHORS,
                                       anchor_masks=ANCHOR_MASKS,
                                       ignore_thresh=IGNORE_THRESH,
                                       num_classes=NUM_CLASSES,
                                       device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ① 写入 TensorBoard
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

            if i % 10 == 0:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{ts}[TRAIN] epoch {epoch}, iter {i}, loss {loss.item():.4f}")

        # ---------- 保存 ----------
        if epoch % 5 == 0 or epoch == MAX_EPOCH - 1:
            torch.save(model.state_dict(), f"yolo_epoch{epoch}.pth")

        # ---------- 验证 ----------
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for i, (img, gt_boxes, gt_labels, _) in enumerate(valid_loader):
                img, gt_boxes, gt_labels = img.to(device), gt_boxes.to(device), gt_labels.to(device)
                outputs = model(img)
                loss = get_loss_multiscale(outputs, gt_boxes, gt_labels,
                                           anchors=ANCHORS,
                                           anchor_masks=ANCHOR_MASKS,
                                           ignore_thresh=IGNORE_THRESH,
                                           num_classes=NUM_CLASSES,
                                           device=device)
                val_loss_epoch += loss.item()

                if i % 1 == 0:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"{ts}[VALID] epoch {epoch}, iter {i}, loss {loss.item():.4f}")
        # ② 写入 TensorBoard（一个 epoch 记录一次验证损失）
        writer.add_scalar("Loss/valid", val_loss_epoch / len(valid_loader), epoch)
        # 回到训练模式
        scheduler.step()
        model.train()

        train_map = evaluate_map(model, train_subset_loader, device)  # 若速度慢，可以传一个子集 loader
        valid_map = evaluate_map(model, valid_loader, device)
        train_map=train_map
        valid_map=valid_map
        # ---------- TensorBoard ----------
        writer.add_scalar("mAP/train", train_map, epoch)
        writer.add_scalar("mAP/valid", valid_map, epoch)
        #---------- 打印 ----------
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{ts}[EVAL ] Epoch {epoch:2d} | "
              f"Train mAP  {train_map:.4f} | "
              f"Valid mAP  {valid_map:.4f}")



