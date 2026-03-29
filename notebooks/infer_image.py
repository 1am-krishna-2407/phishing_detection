import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.image_dataset import PhishingImageDataset
from src.image_model import ImagePhishingModel

# ---------------- CONFIG ----------------
DATA_DIR = "data/phase2/images"
MODEL_PATH = "models/image_model_phase2.pt"
BATCH_SIZE = 16
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# Dataset
dataset = PhishingImageDataset(DATA_DIR, train=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = ImagePhishingModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

rows = []

# =======================
# INFERENCE
# =======================

with torch.no_grad():
    for batch_idx, (images, labels, paths) in enumerate(loader):
        images = images.to(device)

        logits = model(images).squeeze()
        probs = torch.sigmoid(logits)

        for i, p in enumerate(probs):

            # 🔥 FIX: extract filename as ID
            filename = os.path.basename(paths[i])

            rows.append([
                filename,
                float(p.cpu()),
                int(labels[i])
            ])

# =======================
# SAVE
# =======================

df = pd.DataFrame(rows, columns=["id", "prob_img", "label"])

os.makedirs("preds", exist_ok=True)
df.to_csv("preds/image_preds.csv", index=False)

print("✅ image_preds.csv created")
print(df.head())