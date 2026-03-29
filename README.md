# 🛡️ Multimodal Phishing Detection System

A **multi-phase, multimodal phishing detection pipeline** that analyzes **text, images, OCR-extracted content, and URL signals**, and combines them using a **fusion model** for robust and accurate classification.

---

## 🚀 Overview

Phishing attacks have evolved beyond simple text-based scams. This project tackles the problem using a **multi-stage deep learning approach**:

* **Phase 1** → Text-based phishing detection
* **Phase 2** → Image-based phishing detection
* **Phase 2/processed** → OCR + Text understanding


The final system leverages **combined intelligence** from multiple sources to improve detection accuracy.

---

## 🧠 Architecture

```
Text Input ───────────────► Text Encoder (Phase 1)
Image Input ──────────────► Image Model (Phase 2)
Image → OCR ─────────────► OCR Text Encoder (Phase 2.5)

All Outputs ─────────────► Fusion Model (Phase 3)
                          ↓
                    Final Prediction
```

---

## 📁 Project Structure

```
PHISHING DETECTION/
│
├── data/                      # Dataset (not included in repo)
│   ├── phase1/
│   ├── phase2/
│
├── models/                    # Pretrained models (not included)
│   ├── fusion_model_phase3.pt
│   ├── image_model_phase2.pt
│   ├── ocr_text_encoder_phase2_5.pt
│   └── text_encoder_phase1.pt
│
├── notebooks/                 # Training & inference scripts
│   ├── train_text_phase1.py
│   ├── train_image_phase2.py
│   ├── train_ocr_text_phase2_5.py
│   ├── infer_image.py
│   ├── infer_ocr_text.py
│   ├── infer_url_text.py
│   └── final_fusion_infer.py
│
├── plots/                     # Evaluation results
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│
├── preds/                     # Model predictions
│
├── src/                       # Core implementation
│   ├── image_dataset.py
│   ├── image_model.py
│   ├── ocr_extractor.py
│   ├── text_dataset.py
│   └── text_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 📦 Models & Dataset

Due to size constraints, pretrained models and datasets are hosted externally.

👉 **Models (Drive Folder):**
[https://drive.google.com/drive/folders/1JqW3P-Cf-226Z4xd8t2l_EAa7wAmOife](https://drive.google.com/drive/folders/1JqW3P-Cf-226Z4xd8t2l_EAa7wAmOife)

👉 **Dataset (Drive File):**
[https://drive.google.com/file/d/1rUaBTgoYQeQAL8LaYEO0VhOOmZFL7kM1/view?usp=drive_link](https://drive.google.com/file/d/1rUaBTgoYQeQAL8LaYEO0VhOOmZFL7kM1/view?usp=drive_link)

After downloading, place them in:

```
models/
data/
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/phishing-detection.git
cd phishing-detection
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\\Scripts\\activate         # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🧪 Training

Run individual phases:

### Phase 1 (Text)

```
python notebooks/train_text_phase1.py
```

### Phase 2 (Image)

```
python notebooks/train_image_phase2.py
```

### Phase 2.5 (OCR + Text)

```
python notebooks/train_ocr_text_phase2_5.py
```

---

## 🔍 Inference

### Individual modalities

```
python notebooks/infer_image.py
python notebooks/infer_ocr_text.py
python notebooks/infer_url_text.py
```

### Final Fusion Prediction

```
python notebooks/final_fusion_infer.py
```

---

## 📊 Results

The model performance is evaluated using:

* Confusion Matrix
* Precision-Recall Curve
* ROC Curve

Results available in:

```
plots/
```

---

## 🔥 Key Features

* ✅ Multimodal learning (Text + Image + OCR + URL)
* ✅ Modular architecture (phase-wise training)
* ✅ Fusion-based decision making
* ✅ Scalable and extensible design
* ✅ Real-world phishing detection use-case

---

## 🚧 Future Improvements

* API deployment (FastAPI / Django)
* Real-time phishing detection system
* Browser extension integration
* Model optimization for latency
* Deployment on cloud (AWS / GCP)

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed as part of an advanced machine learning project focusing on **cybersecurity and multimodal AI systems**.

---

## ⭐ If you found this useful

Give this repo a ⭐ and share it!
