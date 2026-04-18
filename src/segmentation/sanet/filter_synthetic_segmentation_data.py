import argparse
import json
import os
from collections import defaultdict

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from model import Model


class TrainingDataFilter:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.model.eval()
        self.model.cuda()
        if args.snapshot:
            self.model.load_state_dict(torch.load(args.snapshot), strict=False)
        self.transform = A.Compose([A.Resize(352, 352), A.Normalize(), ToTensorV2()])
        self.load_data()

    def load_data(self):
        with open(self.args.synthetic_jsonl, "r", encoding="utf-8") as f:
            self.synthetic_data = [json.loads(line) for line in f]
        with open(self.args.real_jsonl, "r", encoding="utf-8") as f:
            self.real_data = [json.loads(line) for line in f]
        description_to_real = {item["description"]: item["image"] for item in self.real_data}
        self.valid_samples = []
        for syn_item in self.synthetic_data:
            if syn_item["description"] in description_to_real:
                self.valid_samples.append({
                    "synthetic_image": syn_item["image"],
                    "real_image": description_to_real[syn_item["description"]],
                    "description": syn_item["description"],
                })

    def predict_mask(self, image_tensor, target_shape):
        with torch.no_grad():
            pred = self.model(image_tensor)
            pred = F.interpolate(pred, size=target_shape, mode="bilinear", align_corners=True)[0, 0]
            pos_mask = pred > 0
            neg_mask = pred < 0
            if pos_mask.any():
                pred[pos_mask] /= pos_mask.float().mean()
            if neg_mask.any():
                pred[neg_mask] /= neg_mask.float().mean()
            return torch.sigmoid(pred).cpu().numpy()

    @staticmethod
    def calculate_metrics(pred_mask, gt_mask):
        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (gt_mask > 0.5).astype(np.uint8)
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)
        confidence = pred_mask.mean()
        return {"iou": float(iou), "dice": float(dice), "confidence": float(confidence)}

    @staticmethod
    def calculate_appearance_diversity(synthetic_img, real_img, gt_mask):
        mask_3d = np.stack([gt_mask, gt_mask, gt_mask], axis=2)
        syn_masked = synthetic_img * mask_3d
        real_masked = real_img * mask_3d
        syn_hist = cv2.calcHist([syn_masked.astype(np.uint8)], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        real_hist = cv2.calcHist([real_masked.astype(np.uint8)], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        color_diff = cv2.compareHist(syn_hist, real_hist, cv2.HISTCMP_CHISQR)
        color_diversity = min(color_diff / 1000.0, 1.0)
        syn_gray = cv2.cvtColor(syn_masked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        real_gray = cv2.cvtColor(real_masked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        syn_grad = cv2.Sobel(syn_gray, cv2.CV_64F, 1, 1, ksize=3)
        real_grad = cv2.Sobel(real_gray, cv2.CV_64F, 1, 1, ksize=3)
        texture_diversity = np.abs(syn_grad.mean() - real_grad.mean()) / 255.0
        return {"color_diversity": float(color_diversity), "texture_diversity": float(texture_diversity)}

    @staticmethod
    def training_value(metrics, diversity):
        consistency_score = (metrics["iou"] + metrics["dice"]) / 2
        diversity_score = (diversity["color_diversity"] + diversity["texture_diversity"]) / 2
        return consistency_score * 0.7 + diversity_score * 0.3

    def filter_for_training(self, consistency_threshold=0.7, diversity_threshold=0.1, hard_sample_range=(0.6, 0.85), max_samples_per_category=1450, balance_strategy="mixed"):
        filtered_results = []
        category_counts = defaultdict(int)
        for sample in self.valid_samples:
            synthetic_path = os.path.join(self.args.synthetic_image_dir, sample["synthetic_image"])
            real_path = os.path.join(self.args.real_image_dir, sample["real_image"])
            synthetic_img = cv2.imread(synthetic_path)
            real_img = cv2.imread(real_path)
            if synthetic_img is None or real_img is None:
                continue
            synthetic_img = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2RGB)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            mask_name = os.path.splitext(sample["real_image"])[0] + ".png"
            mask_path = os.path.join(self.args.gt_mask_dir, mask_name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue
            gt_mask = gt_mask / 255.0
            real_h, real_w = real_img.shape[:2]
            synthetic_img = cv2.resize(synthetic_img, (real_w, real_h))
            synthetic_processed = self.transform(image=synthetic_img, mask=gt_mask)
            synthetic_tensor = synthetic_processed["image"].unsqueeze(0).cuda()
            pred_mask = self.predict_mask(synthetic_tensor, (real_h, real_w))
            metrics = self.calculate_metrics(pred_mask, gt_mask)
            diversity = self.calculate_appearance_diversity(synthetic_img, real_img, gt_mask)

            should_include = False
            selection_reason = ""
            if balance_strategy == "consistency":
                if metrics["iou"] >= consistency_threshold and diversity["color_diversity"] >= diversity_threshold:
                    should_include = True
                    selection_reason = "high_consistency"
            elif balance_strategy == "hard":
                if hard_sample_range[0] <= metrics["iou"] <= hard_sample_range[1] and diversity["color_diversity"] >= diversity_threshold:
                    should_include = True
                    selection_reason = "hard_sample"
            else:
                if metrics["iou"] >= consistency_threshold and diversity["color_diversity"] >= diversity_threshold:
                    should_include = True
                    selection_reason = "high_consistency"
                elif hard_sample_range[0] <= metrics["iou"] <= hard_sample_range[1] and diversity["color_diversity"] >= diversity_threshold:
                    should_include = True
                    selection_reason = "hard_sample"

            category = sample["description"][:20]
            if should_include and max_samples_per_category and category_counts[category] >= max_samples_per_category:
                should_include = False
                selection_reason = "category_full"
            if should_include:
                category_counts[category] += 1

            filtered_results.append({
                "synthetic_image": sample["synthetic_image"],
                "real_image": sample["real_image"],
                "description": sample["description"],
                "metrics": metrics,
                "diversity": diversity,
                "selected": should_include,
                "selection_reason": selection_reason,
                "training_value_score": self.training_value(metrics, diversity),
            })
        return filtered_results

    def save_filtered_data(self, filtered_results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        selected_samples = [r for r in filtered_results if r["selected"]]
        with open(os.path.join(output_dir, "filtering_results.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "selected_for_training.jsonl"), "w", encoding="utf-8") as f:
            for sample in selected_samples:
                f.write(json.dumps({"image": sample["synthetic_image"], "description": sample["description"]}, ensure_ascii=False) + "\n")
        with open(os.path.join(output_dir, "selected_images.txt"), "w", encoding="utf-8") as f:
            for sample in selected_samples:
                f.write(f"{sample['synthetic_image']}\n")

        ious = [r["metrics"]["iou"] for r in filtered_results]
        diversities = [r["diversity"]["color_diversity"] for r in filtered_results]
        selected = [r["selected"] for r in filtered_results]
        plt.figure(figsize=(10, 6))
        unselected_idx = [i for i, s in enumerate(selected) if not s]
        selected_idx = [i for i, s in enumerate(selected) if s]
        if unselected_idx:
            plt.scatter([ious[i] for i in unselected_idx], [diversities[i] for i in unselected_idx], c="red", alpha=0.5, s=20, label="rejected")
        if selected_idx:
            plt.scatter([ious[i] for i in selected_idx], [diversities[i] for i in selected_idx], c="green", alpha=0.7, s=20, label="selected")
        plt.xlabel("IoU")
        plt.ylabel("Color diversity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "filtering_visualization.png"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_jsonl", type=str, required=True)
    parser.add_argument("--real_jsonl", type=str, required=True)
    parser.add_argument("--synthetic_image_dir", type=str, required=True)
    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--consistency_threshold", default=0.7, type=float)
    parser.add_argument("--diversity_threshold", default=0.1, type=float)
    parser.add_argument("--balance_strategy", default="mixed", choices=["consistency", "hard", "mixed"])
    parser.add_argument("--max_samples_per_category", default=1450, type=int)
    args = parser.parse_args()

    filter_engine = TrainingDataFilter(Model(args), args)
    results = filter_engine.filter_for_training(
        consistency_threshold=args.consistency_threshold,
        diversity_threshold=args.diversity_threshold,
        balance_strategy=args.balance_strategy,
        max_samples_per_category=args.max_samples_per_category,
    )
    filter_engine.save_filtered_data(results, args.output_dir)
