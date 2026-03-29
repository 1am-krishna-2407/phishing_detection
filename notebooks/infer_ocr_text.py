import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.text_dataset import PhishingTextDataset
from src.text_model import TextPhishingModel

# ---------------- CONFIG ----------------
CSV_PATH = "data/phase2/processed/ocr_text.csv"
MODEL_PATH = "models/ocr_text_encoder_phase2_5.pt"
BATCH_SIZE = 32
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# =======================
# LOAD + FIX DATA
# =======================

df = pd.read_csv(CSV_PATH)

# 🔥 IMPORTANT: match training preprocessing
df["text"] = df["text"].fillna("")
df["text"] = df["text"].apply(
    lambda x: x if len(str(x).strip()) > 0 else "[NO_TEXT]"
)

# Dataset
dataset = PhishingTextDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Model
model = TextPhishingModel().to(device)

# Load encoder weights
model.bert.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

rows = []

# =======================
# INFERENCE
# =======================

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(input_ids, attention_mask).squeeze()
        probs = torch.sigmoid(logits)

        for i, p in enumerate(probs):
            global_idx = batch_idx * BATCH_SIZE + i

            # 🔥 FIX: use original ID instead of index
            sample_id = df.iloc[global_idx]["id"]

            rows.append([
                sample_id,
                float(p.cpu()),
                int(labels[i])
            ])

# =======================
# SAVE
# =======================

df_out = pd.DataFrame(rows, columns=["id", "prob_ocr", "label"])

os.makedirs("preds", exist_ok=True)
df_out.to_csv("preds/ocr_preds.csv", index=False)

print("✅ ocr_preds.csv created")
print(df_out.head())