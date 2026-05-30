---
language:
- en
license: mit
tags:
- image-classification
- skin-cancer
- melanoma
- medical-imaging
- mobilenet
- pytorch
- dermatology
datasets:
- ISIC
metrics:
- accuracy
pipeline_tag: image-classification
---

# 🔬 Skin Cancer Detection — MobileNet-V3 Large

A deep learning model for automated skin lesion classification using **MobileNet-V3 Large** fine-tuned on the ISIC Skin Lesion Dataset. Classifies dermoscopic images into three categories: **Melanoma**, **Nevus**, and **Seborrheic Keratosis**.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNet-V3 Large |
| Validation Accuracy | **83.33%** |
| Classes | 3 (Melanoma, Nevus, Seborrheic Keratosis) |
| Input Size | 224 × 224 px |
| Framework | PyTorch |

---

## 🏷️ Classes

| Label | Type | Description |
|-------|------|-------------|
| **Melanoma** | 🔴 Malignant | Most dangerous form of skin cancer |
| **Nevus** | 🟢 Benign | Common mole (non-cancerous) |
| **Seborrheic Keratosis** | 🟢 Benign | Non-cancerous skin growth |

---

## 🏗️ Model Architecture

- **Base Model:** MobileNet-V3 Large (ImageNet pre-trained)
- **Custom Classifier Head:**
  ```
  Linear(960 → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)
  ```
- **Training Strategy:** Transfer Learning with fine-tuning
- **Dataset:** ISIC (International Skin Imaging Collaboration)

---

## 🚀 How to Use

### Load the Model

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Download model weights
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="janali01/skin-cancer-mobilenetv3",
    filename="checkpoint_mobilenet_v3.pth"
)

# Build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(960, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 3)
)

# Load weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)
```

### Run Inference

```python
CLASS_NAMES = ["Melanoma", "Nevus", "Seborrheic Keratosis"]

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = val_tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]) * 100

label, confidence = predict("skin_lesion.jpg")
print(f"Prediction: {label} ({confidence:.1f}%)")
```

---

## 🌐 Streamlit Web App

A fully functional web app is available on GitHub:

👉 **[GitHub Repository](https://github.com/janali01/skin-cancer-detection)**

### Run Locally

```bash
git clone https://github.com/janali01/skin-cancer-detection
cd skin-cancer-detection
pip install -r requirements.txt
streamlit run app.py
```

### Requirements

```
streamlit
torch
torchvision
Pillow
numpy
huggingface_hub
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `checkpoint_mobilenet_v3.pth` | Trained model weights (PyTorch checkpoint) |
| `history_mobilenet_v3.csv` | Training & validation loss/accuracy per epoch |

---

## ⚠️ Disclaimer

> This model is developed for **educational and research purposes only**. It is **not a substitute for professional medical diagnosis**. Always consult a qualified dermatologist for any skin-related concerns.

---

## 👨‍💻 About

| | |
|-|--|
| **Developer** | Jan Ali |
| **Institution** | SMIU Karachi |
| **Supervisor** | Sir M. Ameen Chhajro |
| **Project Type** | Final Year Project — Computer Vision |
| **Contact** | [Hugging Face Profile](https://huggingface.co/janali01) |

---

## 📜 License

This project is licensed under the **MIT License** — free to use for academic and research purposes.
