import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=348)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--pretrained_backbone", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    opt = parser.parse_args()

    model = PolypPVT(pretrained_path=opt.pretrained_backbone)
    model.load_state_dict(torch.load(opt.pth_path, map_location="cpu"))
    model.cuda()
    model.eval()

    if opt.data_path == "":
        raise ValueError("Please provide --data_path pointing to a benchmark root or a single dataset folder.")

    if os.path.isdir(os.path.join(opt.data_path, "images")):
        datasets = [opt.data_path]
    else:
        datasets = [os.path.join(opt.data_path, name) for name in ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]]

    for data_path in datasets:
        dataset_name = os.path.basename(data_path.rstrip("/"))
        save_path = os.path.join(opt.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        image_root = f"{data_path}/images/"
        gt_root = f"{data_path}/masks/"
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        for _ in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            image = image.cuda()
            p1, p2 = model(image)
            res = F.upsample(p1 + p2, size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(os.path.join(save_path, name), res * 255)
        print(dataset_name, "Finish!")
