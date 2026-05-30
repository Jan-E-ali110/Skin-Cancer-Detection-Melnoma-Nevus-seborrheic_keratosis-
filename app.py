import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 0.5rem;'>
        <h1 style='font-size: 2rem; margin-bottom: 0.2rem;'>🔬 Skin Cancer Detection</h1>
        <p style='color: gray; font-size: 0.95rem;'>Upload a skin lesion image to classify it using AI</p>
    </div>
    <hr style='margin: 0.5rem 0 1.5rem;'>
""", unsafe_allow_html=True)

# ─── Class Labels ──────────────────────────────────────────────────────────────
# ImageFolder alphabetical order: melanoma, nevus, seborrheic_keratosis
CLASS_NAMES = ["Melanoma", "Nevus", "Seborrheic Keratosis"]
POSITIVE_CLASS = "Melanoma"  # malignant

# ─── Model Load ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(960, 256),  
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 3)
    )
    checkpoint = torch.load("checkpoint_mobilenet_v3.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, device

# ─── Transforms ────────────────────────────────────────────────────────────────
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Prediction ────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model, device):
    tensor = val_tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    return pred_label, confidence, probs

# ─── UI ────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG format"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        try:
            model, device = load_model()
            pred_label, confidence, probs = predict(image, model, device)
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()

    st.markdown("### Result")

    is_positive = pred_label == POSITIVE_CLASS

    if is_positive:
        st.error(f"⚠️  **{pred_label}** detected — Please consult a dermatologist.")
        verdict_color = "#c0392b"
        verdict_icon  = "🔴 Positive (Malignant)"
    else:
        st.success(f"✅  **{pred_label}** — No melanoma detected.")
        verdict_color = "#27ae60"
        verdict_icon  = "🟢 Negative (Benign)"

    st.markdown(f"""
        <div style='background:#f8f9fa; border-radius:10px; padding:1rem 1.5rem; margin-top:0.5rem;'>
            <p style='margin:0; font-size:1rem;'>
                <b>Verdict:</b> <span style='color:{verdict_color};'>{verdict_icon}</span>
            </p>
            <p style='margin:0.3rem 0 0; font-size:1rem;'>
                <b>Confidence:</b> {confidence:.1f}%
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Confidence bar for all classes
    st.markdown("#### Confidence Scores")
    for i, cls in enumerate(CLASS_NAMES):
        score = float(probs[i]) * 100
        st.markdown(f"**{cls}**")
        st.progress(score / 100)
        st.caption(f"{score:.1f}%")

    st.markdown("""
        <p style='font-size:0.82rem; color:gray; margin-top:1rem;'>
        ⚠️ This tool is for educational purposes only and does not replace professional medical diagnosis.
        </p>
    """, unsafe_allow_html=True)

else:
    st.info("Please upload a skin lesion image to begin.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; color:gray; font-size:0.85rem; padding-bottom:1rem;'>
        <p style='margin:0;'>Developed by <b>Jan Ali</b> — AI Student, SMIU Karachi</p>
        <p style='margin:0;'>Supervised by <b>Sir M. Ameen Chhajro</b> | Computer Vision Project</p>
        <p style='margin:0; margin-top:0.3rem;'>Model: MobileNet-V3 Large &nbsp;|&nbsp; Dataset: ISIC Skin Lesion &nbsp;|&nbsp; Val Accuracy: 83.33%</p>
    </div>
""", unsafe_allow_html=True)