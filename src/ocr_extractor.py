import os
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

# =======================
# CONFIG
# =======================

# Set your tesseract path (Linux default)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Dataset path
IMAGE_ROOT = "data/phase2/images"
OUTPUT_CSV = "ocr_dataset.csv"


# =======================
# IMAGE PREPROCESSING
# =======================

def preprocess_image(img):
    """Improve image for OCR"""
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8)

    # Accept RGB/BGR arrays and normalize to grayscale for OCR.
    if getattr(img, "ndim", 0) == 3:
        gray = np.mean(img.astype(np.float32), axis=2).astype(np.uint8)
    else:
        gray = np.asarray(img, dtype=np.uint8)

    pil_image = Image.fromarray(gray)
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=1))

    # Use a simple mean-based threshold to increase text contrast
    # without depending on OpenCV in deployment.
    threshold = int(np.asarray(blurred, dtype=np.uint8).mean())
    binary = blurred.point(lambda pixel: 255 if pixel > threshold else 0)
    return np.asarray(binary, dtype=np.uint8)


# =======================
# OCR FUNCTION
# =======================

def extract_text_from_image(image_path):
    """Extract text safely from image"""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        print(f"❌ Failed to load: {image_path}")
        return ""

    try:
        processed = preprocess_image(np.asarray(img))

        # OCR config optimized for blocks of text
        text = pytesseract.image_to_string(processed, config='--psm 6')

        return text.strip()

    except Exception as e:
        print(f"⚠️ OCR failed for {image_path}: {e}")
        return ""


# =======================
# MAIN DATASET CREATION
# =======================

def extract_dataset(image_root, output_csv):
    rows = []

    total_images = 0
    empty_text = 0
    failed_images = 0

    print("📂 Reading from:", os.path.abspath(image_root))

    for label, folder in enumerate(["legit", "phishing"]):
        folder_path = os.path.join(image_root, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue

        print(f"\n📁 Processing: {folder}")

        for img_name in os.listdir(folder_path):

            # Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            total_images += 1

            img_path = os.path.join(folder_path, img_name)

            text = extract_text_from_image(img_path)

            if text == "":
                empty_text += 1

            # ✅ KEEP ALL DATA (important for fusion)
            rows.append({
                "id": img_name,   # critical for merging later
                "text": text,
                "label": label
            })

            # Optional debug (uncomment if needed)
            # print(f"{img_name} → len={len(text)}")

    # Save dataset
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n✅ OCR dataset created:", output_csv)
    print(f"📊 Total images processed: {total_images}")
    print(f"⚠️ Empty OCR text: {empty_text}")
    print(f"❌ Failed images: {failed_images}")
    print(f"📁 Final dataset size: {len(df)}")


# =======================
# RUN SCRIPT
# =======================

if __name__ == "__main__":
    # Quick sanity check
    if not os.path.exists(IMAGE_ROOT):
        print("❌ ERROR: Path not found:", IMAGE_ROOT)
    else:
        print("✅ Path found:", IMAGE_ROOT)
        print("📂 Subfolders:", os.listdir(IMAGE_ROOT))

    extract_dataset(IMAGE_ROOT, OUTPUT_CSV)
    
