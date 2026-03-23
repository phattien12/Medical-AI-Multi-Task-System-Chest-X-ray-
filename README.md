# 🚀 Medical AI Multi-Task System (Chest X-ray)

<p align="center">
  <b>Deep Learning • Computer Vision • Medical Imaging</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-red?logo=pytorch&style=flat-square">
  <img src="https://img.shields.io/badge/Dataset-Chest_Xray-green?style=flat-square">
  <img src="https://img.shields.io/badge/Task-Multi--Task-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Platform-Colab-orange?style=flat-square">
</p>

---

## 📌 Overview

This project builds a **comprehensive Medical AI pipeline** for **Chest X-ray analysis**, covering multiple core computer vision tasks:

### ✨ Tasks Implemented

* 🧠 **Classification** (Pneumonia detection)
* 🧩 **Segmentation** (Pneumothorax mask)
* 🎯 **Object Detection** (Bounding box localization)
* 🧬 **AutoEncoder** (Reconstruction)
* 🎨 **GAN** (Synthetic image generation)
* 🌫️ **Diffusion Model** (Advanced generative AI)

> ⚡ Entire system runs in a **single `.ipynb` file on Google Colab**

---

## ⚙️ Installation

```bash id="inst1"
pip install torch torchvision timm opencv-python matplotlib albumentations
pip install ultralytics
pip install diffusers transformers accelerate
```

---

## 📂 Dataset

### 🏥 Chest X-ray (Pneumonia)

* Source: Kaggle
* Classes: `NORMAL`, `PNEUMONIA`

### 🫁 Pneumothorax Dataset

* Includes:

  * Images
  * Segmentation masks

```bash id="data1"
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks
```

---

## 🧠 Models

### 1️⃣ CNN Classifier

```python id="cnn1"
class CNNClassifier(nn.Module):
```

✔ Detect pneumonia
✔ Lightweight & fast

---

### 2️⃣ Vision Transformer (ViT)

```python id="vit1"
timm.create_model("vit_base_patch16_224", pretrained=True)
```

✔ Transfer learning
✔ Better feature extraction

---

### 3️⃣ U-Net (Segmentation)

```python id="unet1"
class UNet(nn.Module):
```

✔ Segment pneumothorax region
✔ Pixel-level prediction

---

### 4️⃣ Faster R-CNN (Detection)

```python id="det1"
torchvision.models.detection.fasterrcnn_resnet50_fpn()
```

✔ Detect abnormal regions
✔ Bounding box prediction

---

### 5️⃣ AutoEncoder

```python id="ae1"
class AutoEncoder(nn.Module):
```

✔ Image reconstruction
✔ Feature compression

---

### 6️⃣ GAN (Generator + Discriminator)

```python id="gan1"
class Generator(nn.Module):
class Discriminator(nn.Module):
```

✔ Generate synthetic X-ray images

---

### 7️⃣ Diffusion Model

```python id="diff1"
DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
```

✔ State-of-the-art generative model
✔ High-quality image synthesis

---

## 🏋️ Training

### Classification

* Epochs: 5
* Loss: CrossEntropy

### Segmentation

* Loss: BCEWithLogits
* Mixed precision training (AMP)

### Detection

* Model: Faster R-CNN
* Custom dataset with bounding boxes

### Generative Models

* AutoEncoder: MSE Loss
* GAN: Adversarial training
* Diffusion: Pretrained pipeline

---

## 🎨 Results & Visualization

### 🧠 Classification

* Predict pneumonia vs normal

### 🧩 Segmentation

* Ground truth vs predicted mask

### 🎯 Detection

* Bounding boxes on X-ray

### 🧬 Reconstruction

* Original vs reconstructed image

### 🎨 GAN Output

* Synthetic X-ray images

### 🌫️ Diffusion Output

* Generated images from noise

---

## ▶️ How to Run

### ✅ Google Colab (Recommended)

1. Upload notebook
2. Upload `kaggle.json`
3. Run all cells

Done ✅

---

## 📁 Project Structure

```id="struct1"
📦 medical-ai-system
 ┣ 📜 medical_ai.ipynb   # MAIN FILE (run this)
 ┣ 📂 data/              # datasets
 ┣ 📂 runs/              # outputs
 ┗ 📜 README.md
```

---

## 🚀 Use Cases

* 🏥 Medical diagnosis support
* 🧠 AI-assisted radiology
* 🔬 Research in medical imaging
* 🤖 Multi-task deep learning systems

---

## 💡 Key Highlights

* ✅ Multi-task learning in one pipeline
* ✅ Covers **ALL major CV tasks**
* ✅ Uses both **CNN + Transformer + Diffusion**
* ✅ End-to-end system

---

## 🚀 Future Improvements

* Train longer (50+ epochs)
* Use larger models (ViT-L, YOLOv8)
* Add evaluation metrics (Accuracy, IoU, mAP)
* Deploy as web app (Gradio/Streamlit)

---

<p align="center">
🔥 This is a full-stack AI project – perfect for portfolio / internship / research
</p>

<p align="center">
⭐ Star this repo if you found it useful!
</p>
