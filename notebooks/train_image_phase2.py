import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.image_dataset import PhishingImageDataset
from src.image_model import ImagePhishingModel

# ---------------- CONFIG ----------------
DATA_DIR = "data/phase2/images"
BATCH_SIZE = 20
EPOCHS = 14
LR = 1e-4
VAL_SPLIT = 0.35
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# =======================
# DATASET + STRATIFIED SPLIT
# =======================

full_dataset = PhishingImageDataset(DATA_DIR, train=True)

indices = list(range(len(full_dataset)))
labels = [full_dataset.samples[i][1] for i in indices]

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    stratify=labels,
    random_state=42
)

train_ds = Subset(full_dataset, train_idx)
val_ds = Subset(full_dataset, val_idx)

# Control augmentation
train_ds.dataset.train = True
val_ds.dataset.train = False

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =======================
# MODEL
# =======================

model = ImagePhishingModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

# =======================
# TRAINING
# =======================

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for images, labels, _ in train_loader:
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.float().to(device)

        with autocast():
            outputs = model(images).view(-1)   # 🔥 FIXED
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

    # =======================
    # VALIDATION
    # =======================

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).view(-1)   # 🔥 FIXED
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.50).long()

            all_preds.extend(preds.cpu().numpy().flatten())   # 🔥 FIXED
            all_labels.extend(labels.cpu().numpy().flatten()) # 🔥 FIXED

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {np.mean(train_losses):.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# =======================
# SAVE MODEL
# =======================

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/image_model_phase2.pt")

print("\n✅ Image model saved: models/image_model_phase2.pt")