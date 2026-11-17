"""Brain Tumor MRI Classification: single-file Streamlit app (deployable straight from GitHub)."""
import os
import urllib.request
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ------------------------------------------------------------------ settings
HERE = Path(__file__).resolve().parent
MODEL_FILENAME = "best_model.h5"      # the model file you upload next to this app.py
MODEL_URL = ""                        # optional: direct download link (e.g. a GitHub Release file), used only
                                      # if the model file is not found next to this app.py
MODEL_NAME = "ResNet50"
MODEL_URL = "https://github.com/Avani-25/brain-tumor-mri-classification/releases/tag/v1.0"
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.60                 # below this we warn the user

CLASS_INFO = {
    "glioma": "A tumor that arises from glial (supporting) cells of the brain or spinal cord.",
    "meningioma": "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
    "no_tumor": "No tumor pattern was detected by the model in this image.",
    "pituitary": "A tumor in the pituitary gland, located at the base of the brain.",
}

# Test-set results from the project (246 images)
RESULTS = pd.DataFrame({
    "Model": ["Custom CNN", "MobileNetV2", "ResNet50 (deployed)"],
    "Accuracy": [0.663, 0.833, 0.911],
    "Macro F1": [0.659, 0.831, 0.906],
    "Parameters (M)": [1.01, 2.59, 24.11],
    "CPU ms/image": [22.6, 37.1, 68.7],
})
RESNET_CM = pd.DataFrame(
    [[76, 4, 0, 0], [1, 55, 1, 6], [1, 8, 39, 1], [0, 0, 0, 54]],
    index=["True: Glioma", "True: Meningioma", "True: No tumor", "True: Pituitary"],
    columns=["Pred: Glioma", "Pred: Meningioma", "Pred: No tumor", "Pred: Pituitary"],
)

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")


# ------------------------------------------------------------------ helpers
def locate_model():
    """Look for the model file next to this script (or in a models/ folder)."""
    for folder in (HERE, HERE / "models", HERE.parent / "models"):
        for name in (MODEL_FILENAME, "ResNet50.h5"):
            if (folder / name).exists():
                return folder / name
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model once; download it first if a MODEL_URL is given and the file is missing."""
    path = locate_model()
    if path is None and MODEL_URL:
        target = HERE / MODEL_FILENAME
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(target)
        path = target
    if path is None:
        raise FileNotFoundError(
            f"Model file '{MODEL_FILENAME}' not found next to app.py, and no MODEL_URL is set.")
    return keras.models.load_model(path, compile=False)


def preprocess(image):
    """Same preprocessing as training: RGB -> resize to 224x224 -> scale to 0-1."""
    arr = np.array(image.convert("RGB"), dtype="float32")
    arr = tf.image.resize(arr, IMG_SIZE).numpy() / 255.0
    return np.expand_dims(arr, axis=0)


def pretty(name):
    return name.replace("_", " ").title()


# ------------------------------------------------------------------ load model
try:
    with st.spinner("Loading model (the first start can take a minute)..."):
        model = load_model()
except Exception as err:
    st.error("Could not load the model.")
    st.exception(err)
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.write(f"**Model:** {MODEL_NAME} (transfer learning, fine-tuned)")
    st.write("**Classes:**")
    for c in CLASS_NAMES:
        st.write(f"- {pretty(c)}")
    st.warning("Educational project only. This tool is **not** a medical device and must not be used "
               "for diagnosis. Always consult a qualified radiologist or physician.")

# ------------------------------------------------------------------ main page
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
                probs = model.predict(preprocess(image), verbose=0)[0]

            top = int(np.argmax(probs))
            label, conf = CLASS_NAMES[top], float(probs[top])

            st.subheader("Prediction")
            if conf < LOW_CONFIDENCE:
                st.warning(f"**{pretty(label)}** with only {conf:.1%} confidence. "
                           "The model is unsure, so treat this result as unreliable.")
            else:
                st.success(f"**{pretty(label)}**: {conf:.1%} confidence")
            st.write(CLASS_INFO.get(label, ""))

            scores = pd.DataFrame({"Class": [pretty(c) for c in CLASS_NAMES], "Confidence": probs}).set_index("Class")
            st.bar_chart(scores)

            table = scores.copy()
            table["Confidence"] = (table["Confidence"] * 100).round(2).astype(str) + " %"
            st.table(table)

with tab_about:
    st.subheader("Model comparison (test set, 246 images)")
    st.dataframe(RESULTS, hide_index=True)
    st.subheader("ResNet50 confusion matrix (test set)")
    st.dataframe(RESNET_CM)
    st.caption("Only 1 of 197 tumor images was labeled 'no tumor'. "
               "The test set is small, so treat small differences with caution.")
