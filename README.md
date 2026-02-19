# 🌦 Weather Image Recognition via VGG16 Transfer Learning

This project presents a complete deep learning pipeline for classifying weather conditions from images using **VGG16 Transfer Learning**.  
It demonstrates an end-to-end computer vision workflow including data preprocessing, structured dataset splitting, feature extraction, and progressive fine-tuning for performance optimization.

The project follows modern best practices in deep learning training strategies with TensorFlow and Keras.

---

## 📊 Dataset

Weather Image Dataset from Kaggle:

https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data

The dataset contains labeled images representing multiple weather conditions, organized into class-specific folders.

---

## 🧠 Project Pipeline

### 1️⃣ Data Preparation
- Images loaded directly from Google Drive
- Automated label extraction from folder structure
- Image resizing to **224×224**
- VGG16 preprocessing aligned with ImageNet statistics

---

### 2️⃣ Dataset Splitting
- **70% Training**
- **20% Validation**
- **10% Testing**

Ensures robust model generalization and unbiased evaluation.

---

### 3️⃣ Model Architecture

Feature extractor based on **VGG16 pretrained on ImageNet**.

```
Input (224×224×3)
↓
VGG16 Convolutional Base (Frozen → Fine-Tuned)
↓
Global Average Pooling
↓
Batch Normalization
↓
Dense (ReLU Activation)
↓
Dropout Regularization
↓
Softmax Output Layer
```

---

### 4️⃣ Training Strategy

The model is trained using a **two-phase transfer learning approach**.

#### ✅ Phase 1 — Feature Extraction
- VGG16 fully frozen
- Train classifier head only
- Stable convergence
- Fast adaptation to dataset features

#### 🔥 Phase 2 — Fine-Tuning
- Partial unfreezing of VGG layers
- Very low learning rate optimization
- Feature representation refinement
- Improved generalization performance

---

## 📈 Model Performance

### Feature Extraction Phase
- Fast convergence
- Stable validation trend
- Effective initial feature adaptation

![Feature Extraction Performance](assets/Feature%20Extraction%20Results.png)

### Fine-Tuning Phase
- Improved accuracy
- Reduced validation loss
- Better generalization
- Strong representation refinement

![Fine Tuning Performance](assets/Fine%20Tuning%20Results.png)

---

## 🏆 Final Performance Metrics

| Metric | Result |
|---|---|
| Training Loss | Strong reduction |
| Overfitting | Minimal |
| Generalization | Strong |

---

## 📉 Training Dynamics

- Smooth loss convergence
- Consistent validation performance
- No severe overfitting
- Effective regularization using Dropout and Batch Normalization
- Stable transfer learning optimization

---

## 💡 Key Strengths of the Project

✔ Complete end-to-end deep learning pipeline  
✔ Structured data preprocessing workflow  
✔ Professional transfer learning implementation  
✔ Feature extraction + fine-tuning training design  
✔ Strong validation consistency  
✔ Regularization for generalization control  
✔ Visual monitoring of training dynamics  
✔ Scalable and production-ready architecture  

---

## 🛠 Technologies Used

- TensorFlow
- Keras
- Transfer Learning (VGG16)
- Computer Vision
- GPU Training (Google Colab)
- Optimized Data Pipeline

---

## 🚀 How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Update dataset path if necessary.
4. Run cells sequentially.

### Dependencies

```
tensorflow
keras
numpy
matplotlib
pandas
```

---

## 🎯 Applications

- Weather monitoring systems
- Environmental condition recognition
- Smart surveillance systems
- Autonomous perception pipelines
- Climate data classification research

---

## 📌 Conclusion

This project demonstrates the effectiveness of transfer learning in computer vision by leveraging pretrained convolutional neural networks and adapting them to a domain-specific classification task.  
The two-stage training strategy ensures both stability and performance, producing a robust and well-generalized model.

---

## 👤 Author

**Samir Mohamed : AI & Computer Vision Engineer**
