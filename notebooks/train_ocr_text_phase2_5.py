import sys
import os

# ---- Fix import path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------------------

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.text_dataset import PhishingTextDataset
from src.text_model import TextPhishingModel

# -------- CONFIG --------
CSV_PATH = "data/phase2/processed/ocr_text.csv"
BATCH_SIZE = 20
EPOCHS = 10
LR = 2e-5
VAL_SPLIT = 0.2
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# =======================
# LOAD + FIX DATA
# =======================

df = pd.read_csv(CSV_PATH)

# 🔥 Handle empty OCR text (VERY IMPORTANT)
df["text"] = df["text"].fillna("")
df["text"] = df["text"].apply(
    lambda x: x if len(str(x).strip()) > 0 else "[NO_TEXT]"
)

# Debug info
print("📊 Total samples:", len(df))
print("⚠️ Empty text samples:", (df["text"] == "[NO_TEXT]").sum())
print("📊 Label distribution:\n", df["label"].value_counts())

# =======================
# STRATIFIED SPLIT (FIXES MAJOR ISSUE)
# =======================

train_idx, val_idx = train_test_split(
    np.arange(len(df)),
    test_size=VAL_SPLIT,
    stratify=df["label"],
    random_state=42
)

dataset = PhishingTextDataset(CSV_PATH)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =======================
# MODEL
# =======================

model = TextPhishingModel().to(device)

# 🔥 Dynamic class weight (better than fixed 1.3)
num_pos = (df["label"] == 1).sum()
num_neg = (df["label"] == 0).sum()

pos_weight = torch.tensor([num_neg / num_pos]).to(device)
print("⚖️ Using pos_weight:", pos_weight.item())

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = AdamW(model.parameters(), lr=LR)
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

        # 🔥 Ensure float labels for BCE
        labels = batch["label"].float().to(device)

        with autocast():
            outputs = model(input_ids, attention_mask).squeeze()
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
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids, attention_mask).squeeze()
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.4).long().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)

    # 🔥 Prevent F1 crash
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

torch.save(
    model.bert.state_dict(),
    "models/ocr_text_encoder_phase2_5.pt"
)

print("\n✅ OCR-text encoder saved: models/ocr_text_encoder_phase2_5.pt")