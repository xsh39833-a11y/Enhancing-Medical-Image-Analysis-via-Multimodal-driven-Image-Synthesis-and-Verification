import argparse
import ctypes
import datetime
import json
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

libgcc_s = ctypes.CDLL("libgcc_s.so.1")
sys.dont_write_bytecode = True

from model import Model


class MixedData(Dataset):
    def __init__(self, args):
        self.samples = []
        self.sources = [{"jsonl": args.train_jsonl, "image_root": args.image_root, "mask_root": args.mask_root}]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ])
        for src in self.sources:
            self.samples.extend(self._load_jsonl(src))
        self.color1, self.color2 = [], []
        for sample in self.samples:
            if sample["name"][:-4].isdigit():
                self.color1.append(sample)
            else:
                self.color2.append(sample)

    def _load_jsonl(self, src):
        samples = []
        with open(src["jsonl"], "r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                image_name = item["image"]
                image_path = os.path.join(src["image_root"], image_name)
                mask_path = os.path.join(src["mask_root"], image_name)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    samples.append({"name": image_name, "image_path": image_path, "mask_path": mask_path})
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if len(self.color1) == 0:
            ref_sample = self.color2[idx % len(self.color2)]
        elif len(self.color2) == 0:
            ref_sample = self.color1[idx % len(self.color1)]
        else:
            ref_sample = self.color1[idx % len(self.color1)] if np.random.rand() < 0.7 else self.color2[idx % len(self.color2)]
        image2 = cv2.imread(ref_sample["image_path"])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
        mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0, 1), keepdims=True), image2.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1e-6, std)
        image = np.uint8(np.clip((image - mean) / std * std2 + mean2, 0, 255))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)
        return pair["image"], pair["mask"]

    def __len__(self):
        return len(self.samples)


def bce_dice(pred, mask):
    ce_loss = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter / (union + 1)).mean()
    return ce_loss, dice_loss


class Train:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.savepath, exist_ok=True)
        self.data = MixedData(args)
        self.loader = DataLoader(self.data, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        self.model = Model(args)
        self.model.train(True)
        self.model.cuda()
        if args.snapshot is not None:
            self.model.load_state_dict(torch.load(args.snapshot), strict=False)
        base, head = [], []
        for name, param in self.model.named_parameters():
            if "bkbone" in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer = torch.optim.SGD(
            [{"params": base, "lr": 0.1 * args.lr}, {"params": head, "lr": args.lr}],
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        self.scaler = GradScaler()
        self.logger = SummaryWriter(args.savepath)

    def train(self):
        global_step = 0
        for epoch in range(self.args.epoch):
            if epoch + 1 in [64, 96]:
                self.optimizer.param_groups[0]["lr"] *= 0.5
                self.optimizer.param_groups[1]["lr"] *= 0.5
            for image, mask in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                rand = np.random.choice([256, 288, 320, 352], p=[0.1, 0.2, 0.3, 0.4])
                image = F.interpolate(image, size=(rand, rand), mode="bilinear")
                mask = F.interpolate(mask.unsqueeze(1), size=(rand, rand), mode="nearest").squeeze(1)
                with autocast():
                    pred = self.model(image)
                    pred = F.interpolate(pred, size=mask.shape[1:], mode="bilinear", align_corners=True)[:, 0, :, :]
                    loss_ce, loss_dice = bce_dice(pred, mask)
                    total_loss = loss_ce + loss_dice
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                global_step += 1
                self.logger.add_scalar("lr", self.optimizer.param_groups[0]["lr"], global_step=global_step)
                self.logger.add_scalars("loss", {"ce": loss_ce.item(), "dice": loss_dice.item()}, global_step=global_step)
                if global_step % 10 == 0:
                    print(f"{datetime.datetime.now()} | step:{global_step}/{epoch + 1}/{self.args.epoch} | lr={self.optimizer.param_groups[0]['lr']:.6f} | ce={loss_ce.item():.6f} | dice={loss_dice.item():.6f}")
            if (epoch + 1) % 8 == 0:
                save_file = os.path.join(self.args.savepath, f"model-{epoch + 1}")
                torch.save(self.model.state_dict(), save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--savepath", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--snapshot", default=None)
    args = parser.parse_args()
    Train(args).train()
