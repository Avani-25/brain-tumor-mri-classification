"""Streamlit app: Brain Tumor MRI classification (loads the best model chosen in notebook 05)."""
import os
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
keras = tf.keras
from PIL import Image

# ---------------------------------------------------------------- configuration
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOW_CONFIDENCE = 0.60          # below this we warn the user

CLASS_INFO = {
    "glioma": "A tumor that arises from glial (supporting) cells of the brain or spinal cord.",
    "meningioma": "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
    "no_tumor": "No tumor pattern was detected by the model in this image.",
    "pituitary": "A tumor in the pituitary gland, located at the base of the brain.",
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_config():
    """Load the deployment model once and reuse it across reruns."""
    with open(MODELS_DIR / "best_model.json") as f:
        cfg = json.load(f)
    model = keras.models.load_model(MODELS_DIR / cfg["file"], compile=False)
    return model, cfg


def preprocess(image, img_size):
    """Same preprocessing as training: RGB -> resize -> scale to 0-1 (model handles its own normalization)."""
    arr = np.array(image.convert("RGB"), dtype="float32")
    arr = tf.image.resize(arr, img_size).numpy() / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image, img_size):
    return model.predict(preprocess(image, img_size), verbose=0)[0]


def pretty(name):
    return name.replace("_", " ").title()


# ---------------------------------------------------------------- load model
try:
    model, cfg = load_model_and_config()
except Exception as err:
    st.error("Could not load the model. Make sure `models/best_model.h5` and `models/best_model.json` "
             "exist (created in notebook 05).")
    st.exception(err)
    st.stop()

class_names = cfg["class_names"]
img_size = tuple(cfg["img_size"])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write(f"**Model:** {cfg['model_name']} (transfer learning, fine-tuned)")
    st.write("**Classes:**")
    for c in class_names:
        st.write(f"- {pretty(c)}")
    st.warning("Educational project only. This tool is **not** a medical device and must not be used "
               "for diagnosis. Always consult a qualified radiologist or physician.")

# ---------------------------------------------------------------- main page
st.title("🧠 Brain Tumor MRI Classification")
st.caption("Upload a brain MRI image to predict the tumor type with model confidence scores.")

tab_predict, tab_about = st.tabs(["Classify an MRI", "Model performance"])

with tab_predict:
    uploaded = st.file_uploader("Upload a brain MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an MRI image to get a prediction.")
    else:
        image = Image.open(uploaded)
        col_img, col_res = st.columns(2)

        with col_img:
            st.image(image, caption="Uploaded MRI")

        with col_res:
            with st.spinner("Analyzing image..."):
                probs = predict(model, image, img_size)

            top = int(np.argmax(probs))
            label, conf = class_names[top], float(probs[top])

            st.subheader("Prediction")
            if conf < LOW_CONFIDENCE:
                st.warning(f"**{pretty(label)}** with only {conf:.1%} confidence. "
                           "The model is unsure, so treat this result as unreliable.")
            else:
                st.success(f"**{pretty(label)}**: {conf:.1%} confidence")
            st.write(CLASS_INFO.get(label, ""))

            scores = pd.DataFrame({"Class": [pretty(c) for c in class_names], "Confidence": probs}).set_index("Class")
            st.bar_chart(scores)

            table = scores.copy()
            table["Confidence"] = (table["Confidence"] * 100).round(2).astype(str) + " %"
            st.table(table)

with tab_about:
    st.subheader("Model comparison (test set)")
    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    if comparison_csv.exists():
        st.dataframe(pd.read_csv(comparison_csv))
    else:
        st.info("Run notebook 05 to generate the comparison table.")

    cm_img = REPORTS_DIR / "figures" / "05_confusion_matrices.png"
    if cm_img.exists():
        st.image(str(cm_img), caption="Confusion matrices on the test set")



        