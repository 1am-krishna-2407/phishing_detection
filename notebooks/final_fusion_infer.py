import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =======================
# LOAD PREDICTIONS
# =======================

url = pd.read_csv("preds/url_preds.csv")
img = pd.read_csv("preds/image_preds.csv")
ocr = pd.read_csv("preds/ocr_preds.csv")

print("📊 Sizes:")
print("URL:", len(url))
print("IMG:", len(img))
print("OCR:", len(ocr))

# =======================
# NORMALIZE IDS
# =======================

url["id"] = url["id"].astype(str)
img["id"] = img["id"].astype(str)
ocr["id"] = ocr["id"].astype(str)

# =======================
# REMOVE DUPLICATES
# =======================

img = img.drop_duplicates(subset="id", keep="first")
ocr = ocr.drop_duplicates(subset="id", keep="first")
url = url.drop_duplicates(subset="id", keep="first")

print("\n📊 After removing duplicates:")
print("IMG:", len(img))
print("OCR:", len(ocr))
print("URL:", len(url))

# =======================
# MERGE IMAGE + OCR
# =======================

df = img.merge(ocr, on="id", how="inner")

print("\n📊 Image+OCR merged:", len(df))

# =======================
# ADD URL (OPTIONAL)
# =======================

if "prob_url" not in url.columns:
    if "prob" in url.columns:
        url = url.rename(columns={"prob": "prob_url"})
    else:
        print("⚠️ No URL probabilities found")

df = df.merge(url[["id", "prob_url"]], on="id", how="left")

print("\n📊 After adding URL:", len(df))

# =======================
# FIX LABEL COLUMN
# =======================

label_cols = [col for col in df.columns if "label" in col]

if len(label_cols) > 1:
    df["label"] = df[label_cols[0]]
    df = df.drop(columns=label_cols[1:])
elif len(label_cols) == 1:
    df["label"] = df[label_cols[0]]

# =======================
# FUSION FUNCTION (UPDATED WEIGHTS)
# =======================

def fuse(prob_url=None, prob_img=None, prob_ocr=None):
    weights = {
        "url": 0.35,
        "img": 0.45,
        "ocr": 0.20
    }

    score = 0
    total = 0

    if pd.notna(prob_url):
        score += weights["url"] * prob_url
        total += weights["url"]

    if pd.notna(prob_img):
        score += weights["img"] * prob_img
        total += weights["img"]

    if pd.notna(prob_ocr):
        score += weights["ocr"] * prob_ocr
        total += weights["ocr"]

    return score / total if total > 0 else None

# =======================
# THRESHOLD RULE (STRONG TRIGGERS ONLY)
# =======================

def threshold_rule(row):

    # Very high confidence only
    if pd.notna(row.get("prob_url")) and row["prob_url"] > 0.90:
        return 1

    if row["prob_img"] > 0.90:
        return 1

    if row["prob_ocr"] > 0.80:
        return 1

    return 0

# =======================
# APPLY LOGIC
# =======================

final_scores = []
predictions = []

for _, row in df.iterrows():

    # Step 1: Strong trigger
    if threshold_rule(row):
        final_scores.append(1.0)
        predictions.append(1)
        continue

    # Step 2: Weighted fusion
    score = fuse(
        row.get("prob_url"),
        row.get("prob_img"),
        row.get("prob_ocr")
    )

    final_scores.append(score)

    # Step 3: Higher cutoff (handles imbalance)
    predictions.append(1 if score and score > 0.60 else 0)

df["final_score"] = final_scores
df["prediction"] = predictions

# =======================
# SAVE
# =======================

os.makedirs("preds", exist_ok=True)
df.to_csv("preds/final_predictions.csv", index=False)

print("\n✅ Final predictions saved")
print(df.head())

# =======================
# EVALUATION
# =======================

if "label" in df.columns:
    y_true = df["label"]
    y_pred = df["prediction"]

    print("\n📊 Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# =======================
# FINAL STATS
# =======================

print("\n📊 Final size:", len(df))
print("\n📊 Prediction distribution:")
print(df["prediction"].value_counts())


os.makedirs("plots", exist_ok=True)

if "label" in df.columns:

    y_true = df["label"]
    y_pred = df["prediction"]
    y_scores = df["final_score"]

    # -----------------------
    # 1. CONFUSION MATRIX
    # -----------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # -----------------------
    # 2. ROC-AUC CURVE
    # -----------------------
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.close()

    # -----------------------
    # 3. PRECISION-RECALL CURVE
    # -----------------------
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("plots/precision_recall_curve.png")
    plt.close()

print("\n📊 Plots saved in /plots folder")