import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.text_dataset import PhishingTextDataset
from src.text_model import TextPhishingModel

# ---------------- CONFIG ----------------
CSV_PATH = "data/phase1/processed/urls_phase1.csv"
MODEL_PATH = "models/text_encoder_phase1.pt"
BATCH_SIZE = 32
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# =======================
# LOAD DATA
# =======================

df = pd.read_csv(CSV_PATH)

# ✅ FIX 1: Auto-create ID if missing
if "id" not in df.columns:
    print("⚠️ 'id' column not found. Creating one...")
    df.insert(0, "id", range(len(df)))

# Dataset
dataset = PhishingTextDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ FIX 2: Safety check
if len(dataset) != len(df):
    raise ValueError(f"❌ Dataset size ({len(dataset)}) != CSV size ({len(df)})")

# Model
model = TextPhishingModel().to(device)
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

            # ✅ FIX 3: Prevent overflow indexing
            if global_idx >= len(df):
                continue

            sample_id = df.iloc[global_idx]["id"]

            rows.append([
                sample_id,
                float(p.cpu()),
                int(labels[i])
            ])

# =======================
# SAVE
# =======================

df_out = pd.DataFrame(rows, columns=["id", "prob_url", "label"])

os.makedirs("preds", exist_ok=True)
df_out.to_csv("preds/url_preds.csv", index=False)

print("✅ url_preds.csv created")
print(df_out.head())