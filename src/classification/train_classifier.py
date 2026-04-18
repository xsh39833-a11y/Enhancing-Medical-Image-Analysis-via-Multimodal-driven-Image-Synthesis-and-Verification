"""Multimodal classifier training for skin disease diagnosis.

This release version is adapted from the author's experimental training script
and keeps the original image-text classifier design while exposing a clean CLI.
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_label_mapping(json_files):
    all_labels = set()
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle]
            all_labels.update(item["label"] for item in data)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class MultiModalDataset(Dataset):
    def __init__(self, json_file, image_dir, label_to_idx, transform=None):
        with open(json_file, "r", encoding="utf-8") as handle:
            self.data = [json.loads(line) for line in handle]
        self.image_dir = image_dir
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[item["label"]]
        text = item["description"]
        return image, text, label


class MultiModalClassifier(nn.Module):
    def __init__(self, text_encoder_path, num_classes=3, lambda_mse=0.75):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_feat_dim = 512
        self.lambda_mse = lambda_mse
        self.mse_fn = nn.MSELoss()

        self.text_encoder = BertModel.from_pretrained(text_encoder_path)
        self.text_feat_dim = 768
        self.projection_matrix = nn.Parameter(torch.randn(self.text_feat_dim, self.img_feat_dim))

        for name, param in self.text_encoder.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in range(11)):
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.img_encoder(image).squeeze(-1).squeeze(-1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        text_proj_feat = torch.matmul(text_feat, self.projection_matrix)
        mse_loss = self.mse_fn(text_proj_feat, img_feat)
        combined = torch.cat([img_feat, text_proj_feat], dim=1)
        logits = self.classifier(combined)
        return logits, mse_loss * self.lambda_mse


def evaluate(model, loader, tokenizer, criterion, device, num_classes):
    model.eval()
    total_loss, total_preds, total_labels = 0, [], []
    all_probs = []
    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            texts = [str(t) for t in texts]
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            logits, weighted_mse = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels) + weighted_mse

            total_loss += loss.item() * images.size(0)
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(total_labels, total_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    y_true = np.eye(num_classes)[np.array(total_labels)]
    try:
        auc = roc_auc_score(y_true, all_probs, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc


def main(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    label_to_idx, idx_to_label = create_label_mapping([args.train_jsonl, args.val_jsonl])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultiModalDataset(args.train_jsonl, args.image_dir, label_to_idx, transform=train_transform)
    val_dataset = MultiModalDataset(args.val_jsonl, args.image_dir, label_to_idx, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder_path)
    num_classes = len(label_to_idx)
    model = MultiModalClassifier(args.text_encoder_path, num_classes=num_classes, lambda_mse=args.lambda_mse).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_score = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_preds, total_labels = 0, [], []
        for images, texts, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            texts = [str(t) for t in texts]
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            optimizer.zero_grad()
            logits, weighted_mse = model(images, input_ids, attention_mask)
            ce_loss = criterion(logits, labels)
            loss = ce_loss + weighted_mse
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(total_labels, total_preds)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, tokenizer, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )

        score = 2 * (val_acc * val_auc) / (val_acc + val_auc) if (val_acc + val_auc) > 0 else 0
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, "multimodal_model.pth"))

    with open(os.path.join(args.save_dir, "label_mapping.json"), "w", encoding="utf-8") as handle:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, handle, indent=2)

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multimodal classifier.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_mse", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
