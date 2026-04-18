"""Standalone evaluation for the multimodal classifier."""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer

from train_classifier import MultiModalClassifier, MultiModalDataset, create_label_mapping, set_seed


def main(args):
    set_seed(args.seed)
    label_to_idx, _ = create_label_mapping([args.train_jsonl, args.test_jsonl])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = MultiModalDataset(args.test_jsonl, args.image_dir, label_to_idx, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder_path)
    model = MultiModalClassifier(args.text_encoder_path, num_classes=len(label_to_idx), lambda_mse=args.lambda_mse).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, total_preds, total_labels = 0, [], []
    all_probs = []
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            encoded = tokenizer([str(t) for t in texts], padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            logits, weighted_mse = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels) + weighted_mse

            total_loss += loss.item() * images.size(0)
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    y_true = np.eye(len(label_to_idx))[np.array(total_labels)]
    try:
        auc = roc_auc_score(y_true, all_probs, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    metrics = {
        "test_loss": total_loss / len(test_loader.dataset),
        "accuracy": float(acc),
        "auc": float(auc),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the multimodal classifier.")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Used only for consistent label mapping.")
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_mse", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
