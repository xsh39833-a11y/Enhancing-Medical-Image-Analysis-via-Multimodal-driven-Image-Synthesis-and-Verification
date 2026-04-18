"""Task-aware filtering for synthetic classification samples."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from torchvision import transforms
from transformers import BertTokenizer

from train_classifier import MultiModalClassifier, create_label_mapping, set_seed


class SyntheticDatasetClassifier:
    def __init__(self, model, tokenizer, device, label_to_idx, idx_to_label):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label

    def classify_synthetic_data(self, synthetic_jsonl_path, synthetic_image_dir, high_conf_threshold=0.8, low_conf_threshold=0.6):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(synthetic_jsonl_path, "r", encoding="utf-8") as handle:
            synthetic_data = [json.loads(line) for line in handle]

        self.model.eval()
        all_predictions = []
        high_conf_data = []
        low_conf_data = []
        high_conf_wrong_data = []

        with torch.no_grad():
            for item in synthetic_data:
                image_path = os.path.join(synthetic_image_dir, item["image"])
                if not os.path.exists(image_path):
                    continue

                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                encoded = self.tokenizer([str(item["description"])], padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                logits, _ = self.model(image_tensor, input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probs, dim=1)

                confidence = confidence.item()
                predicted_label = self.idx_to_label[predicted_idx.item()]
                true_label = item["label"]

                prediction_info = {
                    "image": item["image"],
                    "description": item["description"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct": predicted_label == true_label,
                }
                all_predictions.append(prediction_info)

                if confidence >= high_conf_threshold and predicted_label == true_label:
                    high_conf_data.append({
                        "image": item["image"],
                        "description": item["description"],
                        "label": predicted_label,
                        "confidence": confidence,
                    })
                elif confidence >= high_conf_threshold and predicted_label != true_label:
                    high_conf_wrong_data.append({
                        "image": item["image"],
                        "description": item["description"],
                        "label": true_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "adversarial": True,
                    })
                elif confidence <= low_conf_threshold:
                    low_conf_data.append({
                        "image": item["image"],
                        "description": item["description"],
                        "label": true_label,
                        "confidence": confidence,
                        "predicted_label": predicted_label,
                    })

        return high_conf_data, low_conf_data, high_conf_wrong_data, all_predictions

    def save_filtered_data(self, high_conf_data, low_conf_data, high_conf_wrong_data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "high_confidence": os.path.join(output_dir, "high_confidence.jsonl"),
            "low_confidence": os.path.join(output_dir, "low_confidence.jsonl"),
            "high_conf_wrong": os.path.join(output_dir, "high_conf_wrong.jsonl"),
        }
        for key, data in [
            ("high_confidence", high_conf_data),
            ("low_confidence", low_conf_data),
            ("high_conf_wrong", high_conf_wrong_data),
        ]:
            with open(paths[key], "w", encoding="utf-8") as handle:
                for item in data:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return paths


def merge_datasets(original_jsonl, high_conf_jsonl, high_conf_wrong_jsonl, output_jsonl, max_synthetic_ratio=0.3, adversarial_weight=0.1):
    with open(original_jsonl, "r", encoding="utf-8") as handle:
        original_data = [json.loads(line) for line in handle]
    with open(high_conf_jsonl, "r", encoding="utf-8") as handle:
        high_conf_data = [json.loads(line) for line in handle]
    if os.path.exists(high_conf_wrong_jsonl):
        with open(high_conf_wrong_jsonl, "r", encoding="utf-8") as handle:
            high_conf_wrong_data = [json.loads(line) for line in handle]
    else:
        high_conf_wrong_data = []

    max_synthetic_count = int(len(original_data) * max_synthetic_ratio / (1 - max_synthetic_ratio))
    max_adversarial_count = int(max_synthetic_count * adversarial_weight)
    max_positive_count = max_synthetic_count - max_adversarial_count

    high_conf_data = sorted(high_conf_data, key=lambda x: x["confidence"], reverse=True)[:max_positive_count]
    high_conf_wrong_data = sorted(high_conf_wrong_data, key=lambda x: x["confidence"], reverse=True)[:max_adversarial_count]
    merged_data = original_data + high_conf_data + high_conf_wrong_data

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for item in merged_data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_jsonl


def main(args):
    set_seed(args.seed)
    label_to_idx, idx_to_label = create_label_mapping([args.original_train_jsonl])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder_path)
    model = MultiModalClassifier(args.text_encoder_path, num_classes=len(label_to_idx), lambda_mse=args.lambda_mse)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    classifier = SyntheticDatasetClassifier(model, tokenizer, device, label_to_idx, idx_to_label)
    high_conf_data, low_conf_data, high_conf_wrong_data, all_predictions = classifier.classify_synthetic_data(
        args.synthetic_jsonl,
        args.synthetic_image_dir,
        args.high_conf_threshold,
        args.low_conf_threshold,
    )
    saved_paths = classifier.save_filtered_data(high_conf_data, low_conf_data, high_conf_wrong_data, args.output_dir)

    merged_train_path = os.path.join(args.output_dir, "merged_train.jsonl")
    merge_datasets(
        args.original_train_jsonl,
        saved_paths["high_confidence"],
        saved_paths["high_conf_wrong"],
        merged_train_path,
        max_synthetic_ratio=args.max_synthetic_ratio,
        adversarial_weight=args.adversarial_weight,
    )

    accuracy = np.mean([p["correct"] for p in all_predictions]) if all_predictions else 0.0
    report = {
        "num_predictions": len(all_predictions),
        "num_high_confidence": len(high_conf_data),
        "num_low_confidence": len(low_conf_data),
        "num_high_conf_wrong": len(high_conf_wrong_data),
        "accuracy": float(accuracy),
        "merged_train_jsonl": merged_train_path,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    confidences = [p["confidence"] for p in all_predictions]
    if confidences:
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=30)
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title("Confidence Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confidence_distribution.png"))

    labels_true = [p["true_label"] for p in all_predictions]
    labels_pred = [p["predicted_label"] for p in all_predictions]
    if labels_true and labels_pred:
        report_text = classification_report(labels_true, labels_pred)
        with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as handle:
            handle.write(report_text)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter synthetic classification data.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--synthetic_jsonl", type=str, required=True)
    parser.add_argument("--synthetic_image_dir", type=str, required=True)
    parser.add_argument("--original_train_jsonl", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--high_conf_threshold", type=float, default=0.8)
    parser.add_argument("--low_conf_threshold", type=float, default=0.6)
    parser.add_argument("--max_synthetic_ratio", type=float, default=0.3)
    parser.add_argument("--adversarial_weight", type=float, default=0.1)
    parser.add_argument("--lambda_mse", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
