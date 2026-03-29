import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.text_dataset import PhishingTextDataset
from src.text_model import TextPhishingModel

# ---------------- CONFIG ----------------
CSV_PATH = "data/phase1/processed/urls_phase1.csv"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
VAL_SPLIT = 0.2
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# =======================
# DATASET + STRATIFIED SPLIT
# =======================

dataset = PhishingTextDataset(CSV_PATH)

indices = list(range(len(dataset)))
labels = dataset.data["label"].tolist()

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    stratify=labels,
    random_state=42
)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =======================
# MODEL
# =======================

model = TextPhishingModel().to(device)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = GradScaler()

# =======================
# TRAINING
# =======================

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].float().to(device)

        with autocast():
            outputs = model(input_ids, attention_mask).view(-1)   # 🔥 FIXED
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
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask).view(-1)   # 🔥 FIXED
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())       # 🔥 FIXED
            all_labels.extend(labels.cpu().numpy().flatten())     # 🔥 FIXED

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)         # 🔥 FIXED
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
torch.save(model.bert.state_dict(), "models/text_encoder_phase1.pt")

print("\n✅ Text encoder saved: models/text_encoder_phase1.pt")