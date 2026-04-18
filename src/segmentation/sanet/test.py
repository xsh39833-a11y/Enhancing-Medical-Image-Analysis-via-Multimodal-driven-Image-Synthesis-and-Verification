import argparse
import ctypes
import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

libgcc_s = ctypes.CDLL("libgcc_s.so.1")
sys.dont_write_bytecode = True

from model import Model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Data(Dataset):
    def __init__(self, args):
        self.args = args
        self.samples = sorted([name for name in os.listdir(os.path.join(args.datapath, "images")) if not name.startswith(".")])
        self.transform = A.Compose([A.Resize(352, 352), A.Normalize(), ToTensorV2()])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image_path = os.path.join(self.args.datapath, "images", name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        pair = self.transform(image=image)
        return pair["image"], h, w, name

    def __len__(self):
        return len(self.samples)


class Test:
    def __init__(self, args):
        self.args = args
        self.data = Data(args)
        self.loader = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=False, num_workers=args.num_workers)
        self.model = Model(args)
        ckpt = torch.load(args.snapshot, map_location="cpu")
        try:
            self.model.load_state_dict(ckpt, strict=False)
        except RuntimeError:
            new_ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
            self.model.load_state_dict(new_ckpt, strict=False)
        self.model.cuda()
        self.model.eval()

    def save_prediction(self):
        os.makedirs(self.args.predpath, exist_ok=True)
        with torch.no_grad():
            for image, h, w, name in self.loader:
                image = image.cuda().float()
                pred = self.model(image)
                pred = F.interpolate(pred, size=(int(h.item()), int(w.item())), mode="bilinear", align_corners=True)[0, 0]
                pos_mask = pred > 0
                neg_mask = pred < 0
                if pos_mask.any():
                    pred[pos_mask] /= pos_mask.float().mean()
                if neg_mask.any():
                    pred[neg_mask] /= neg_mask.float().mean()
                pred = torch.sigmoid(pred).cpu().numpy()
                pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(self.args.predpath, name[0]), pred)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--predpath", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--snapshot", type=str, required=True)
    args = parser.parse_args()
    Test(args).save_prediction()
