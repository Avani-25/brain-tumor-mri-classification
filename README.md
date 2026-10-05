# **🧠 Brain Tumor MRI Image Classification**

Classify brain MRI scans into 4 categories with deep learning, and try it live in a Streamlit app.

# 📌 Overview

This project builds a custom CNN from scratch and compares it with transfer-learning models (MobileNetV2, EfficientNetB0, ResNet50) to classify brain MRI images into:

# Class                                               	Description
🟠 Glioma                                   	Tumor of the glial (supporting) cells of the brain
🟡 Meningioma	                                Tumor of the membranes surrounding the brain
🟢 No                                         Tumor	No tumor pattern detected
🔵 Pituitary	                                Tumor of the pituitary gland

The best model (ResNet50) is deployed in a Streamlit web app that returns the predicted tumor type with confidence scores.

# 🏆 Results

Evaluated on the held-out test set (246 images):

 Model                        	Accuracy	           Macro             F1	Parameters	                      CPU time / image

1.Custom CNN	            |         66.3%	            0.659	                1.0 M                              	23 ms
2.MobileNetV2	            |         83.3%	            0.831	                2.6 M	                              37 ms
3.EfficientNetB0	        |          —	                  —	                  4.4 M	                                —
4.ResNet50 ⭐            |        91.1%	              0.906	                24.1 M	                            69 ms

 **Highlights**

1.Transfer learning beat the custom CNN by 17 to 25 accuracy points.
2.ResNet50 is the most accurate and reliable model, and it was chosen for deployment.
3.Confusion matrices, training curves and the full comparison table are in reports/.

# 📂 Dataset

Labeled MRI Brain Tumor Dataset from Roboflow Universe (License: CC BY 4.0).

1. 2,443 images (640×640 RGB): 1,695 train, 502 validation, 246 test
2. 4 classes: glioma, meningioma, no tumor, pituitary

  # 🔬 Method
  
1. Explore the data: class balance, resolution, samples
2. Preprocess: resize to 224×224, scale pixels to 0-1
3. Augment (training only): flips, rotation, zoom, shifts, brightness
4. Custom CNN: 5 conv blocks with BatchNorm and Dropout
5. Transfer learning: ImageNet backbones with a new head, trained in two phases (frozen backbone, then fine-tuning the top layers)
6. Train with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau and class weights
7. Evaluate: accuracy, precision, recall, F1, confusion matrix, training curves
8. Compare models and deploy the best one with Streamlit

   # 🚀 Quick Start

   # 1. Clone the repo (needs Git LFS for the model files)
      
git clone https://github.com/YOUR_USERNAME/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
git lfs pull


 # 2. Create and activate a virtual environment
```bash
python -m venv venv
```
venv\Scripts\activate          # Windows
#source venv/bin/activate     # Mac / Linux


# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Run the app
```bash
streamlit run app/app.py
```

# ⚠️ Limitations

1. The test set is small (246 images), so differences of 1-2 points are not significant.
2. Trained and tested on a single public dataset, with no external validation.
3. Not clinically validated.
