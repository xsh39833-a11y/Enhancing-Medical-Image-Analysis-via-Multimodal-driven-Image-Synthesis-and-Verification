import argparse
import json
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from utils.utils import AvgMeter, adjust_lr, clip_gradient


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class MixedPolypDataset(Dataset):
    def __init__(self, sources, trainsize=352, augmentation=False):
        self.trainsize = trainsize
        self.augmentation = augmentation
        self.samples = []
        for src in sources:
            self.samples.extend(self._load_jsonl(src))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])

    def _load_jsonl(self, src):
        samples = []
        with open(src["jsonl"], "r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                image_name = item["image"]
                image_path = os.path.join(src["image_root"], image_name)
                mask_path = os.path.join(src["mask_root"], image_name)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    samples.append({"image_path": image_path, "mask_path": mask_path})
        return samples

    def _random_aug(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        return image, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")
        if self.augmentation:
            image, mask = self._random_aug(image, mask)
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask


def get_mixed_loader(sources, batchsize, trainsize, augmentation=False, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MixedPolypDataset(sources=sources, trainsize=trainsize, augmentation=augmentation)
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = f"{data_path}/images/"
    gt_root = f"{data_path}/masks/"
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    dsc = 0.0
    for _ in range(num1):
        image, gt, _name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= gt.max() + 1e-8
        image = image.cuda()
        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input_arr = res
        target = np.array(gt)
        smooth = 1
        intersection = np.reshape(input_arr, (-1)) * np.reshape(target, (-1))
        dice = (2 * intersection.sum() + smooth) / (input_arr.sum() + target.sum() + smooth)
        dsc += float(f"{dice:.4f}")
    return dsc / num1


def train(train_loader, model, optimizer, epoch, opt, total_step, best, dict_plot):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_p2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
            p1, p2 = model(images)
            loss_p1 = structure_loss(p1, gts)
            loss_p2 = structure_loss(p2, gts)
            loss = loss_p1 + loss_p2
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_p2_record.update(loss_p2.data, opt.batchsize)
        if i % 20 == 0 or i == total_step:
            print(f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], lateral-5: {loss_p2_record.show():0.4f}]")

    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(opt.train_save, f"{epoch}PolypPVT.pth"))

    mean = {}
    for dataset in ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]:
        dataset_dice = test(model, opt.test_path, dataset)
        logging.info("epoch: %s, dataset: %s, dice: %s", epoch, dataset, dataset_dice)
        dict_plot[dataset].append(dataset_dice)
        mean[dataset] = dataset_dice

    meandice = (mean["CVC-300"] * 60 + mean["CVC-ClinicDB"] * 62 + mean["Kvasir"] * 100 + mean["CVC-ColonDB"] * 380 + mean["ETIS-LaribPolypDB"] * 196) / (60 + 62 + 100 + 380 + 196)
    dict_plot["test"].append(meandice)
    if meandice > best:
        best = meandice
        torch.save(model.state_dict(), os.path.join(opt.train_save, "PolypPVT.pth"))
        torch.save(model.state_dict(), os.path.join(opt.train_save, f"{epoch}PolypPVT-best.pth"))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--train_image_root", type=str, required=True)
    parser.add_argument("--train_mask_root", type=str, required=True)
    parser.add_argument("--pretrained_backbone", type=str, default=None)
    parser.add_argument("--train_save", type=str, required=True)
    opt = parser.parse_args()

    logging.basicConfig(filename=os.path.join(opt.train_save, "train_log.log"), level=logging.INFO)
    os.makedirs(opt.train_save, exist_ok=True)

    dict_plot = {"CVC-300": [], "CVC-ClinicDB": [], "Kvasir": [], "CVC-ColonDB": [], "ETIS-LaribPolypDB": [], "test": []}
    model = PolypPVT(pretrained_path=opt.pretrained_backbone).cuda()
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4) if opt.optimizer == "AdamW" else torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    train_sources = [{"jsonl": opt.train_jsonl, "image_root": opt.train_image_root, "mask_root": opt.train_mask_root}]
    train_loader = get_mixed_loader(train_sources, opt.batchsize, opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)
    best = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        best = train(train_loader, model, optimizer, epoch, opt, total_step, best, dict_plot)

    plt.figure()
    plt.plot(dict_plot["test"])
    plt.xlabel("epoch")
    plt.ylabel("mean dice")
    plt.title("Polyp-PVT validation")
    plt.savefig(os.path.join(opt.train_save, "eval.png"))
