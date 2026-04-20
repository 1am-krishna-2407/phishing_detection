import os
import cv2
import pytesseract
import pandas as pd

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (better than fixed)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


# =======================
# OCR FUNCTION
# =======================

def extract_text_from_image(image_path):
    """Extract text safely from image"""
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Failed to load: {image_path}")
        return ""

    try:
        processed = preprocess_image(img)

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
    