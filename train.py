import datetime

import torch
import json
from torch.utils import data
import os
import transforms
import numpy as np
import transforms
import argparse
from model import HRnet
from my_dataset import Cocokeypoints

def main(args):
    fixed_size = args.fixed_size
    heatmap_size = (fixed_size[0]//4, fixed_size[1]//4)
    kps_weights = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((17, 1))

    if args.num_joints > 17:
        for i in range(len(args.num_joints-17)):
            kps_weights = np.append(kps_weights, 1.0)

    upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    lower_body_ids = (11, 12, 13, 14, 15, 16)
    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                       [9, 10], [11, 12], [13, 14], [15, 16]]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transforms = {
        "train": transforms.Compose([
            transforms.HalfBody(0.3, upper_body_ids, lower_body_ids),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, flip_pairs),
            transforms.KeypointsToHeatMap(heatmap_hw=heatmap_size, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    data_root = args.data_path
    batch_size = args.batch_size
    train_dataset = Cocokeypoints(data_root, "train", data_transforms["train"], fixed_size=args.fixed_size)
    val_dataset = Cocokeypoints(data_root, "val", data_transforms["val"], fixed_size=args.fixed_size)

    train_dataset = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_dataset = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=val_collate_fn)

    model = HRnet(32, args.num_joints)
    weight_dict = torch.load("./hrnet_w32.pth", map_location='cpu')
    for k in list(weight_dict.keys()):
        if ("fc" in k) or ("head" in k):
            del weight_dict[k]
        if "final_layer" in k:
            if weight_dict[k].shape[0] != args.num_joints:
                del weight_dict[k]
    model.load_state_dict(weight_dict)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    start_epoch = 0
    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    learing_rate = []
    training_loss = []
    val_map = []
    
    result_file = "result{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    for epoch in range(start_epoch, 210):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_dataset, device=device, epoch=epoch, scaler=scaler)
        training_loss.append(mean_loss.item())
        learing_rate.append(lr)
        lr_scheduler.step()
        
        coco_info = utils.evaluate(model, val_dataset, device=device, epoch=epoch, flip=True, flip_pairs=flip_pairs)
        with open(result_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(coco_info[1])
        save_files={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
    torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_joints", default=17, type=int, help="num_joints")
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument("--batch_size", default=32, type=int)
    #使用混合精度
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp for trainning")
    parser.add_argument("--epochs", default=210, type=int)
    parser.add_argument("--image_size", default=[256, 192], type=int)
    parser.add_argument("--data_path", default="./data/images/train2017")
    parser.add_argument("__output_dir", default="./file_save")
    parser.add_argument("--fixed_size=", default=[256, 192], type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr-steps", default=[170, 200], type=int)
    parser.add_argument("--lr-gramma", default=0.01, type=float)
    parser.add_argument("--resume_path", default="", type=str, help="resume last checkpoint")

    args = parser.parse_args()
    print("Arg is: ", arg)

    if not os.path.exists(args.output_dir):
        os.makedirs(arg.output_dir)

    main(args)