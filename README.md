# 🚗 Vehicle Detection with YOLOv11

This project demonstrates real-time **vehicle detection and classification** using the latest YOLOv11 object detection model. The goal is to detect **Cars, Buses, Ambulances, Motorcycles, and Trucks** using a custom YOLO-formatted dataset developed by [Alkan Erturan](https://www.kaggle.com/datasets/alkanerturan/vehicledetection/data).

---

## 🛠️ Used Technologies

- 🧠 **YOLOv11** – for object detection  
- 🐍 **Python 3.11** – core programming language  
- ⚙️ **PyTorch** – deep learning backend  
- 🚀 **Ultralytics** – YOLO training and deployment framework  
- 📊 **Matplotlib & Seaborn** – result visualization  
- 📁 **OpenCV** – image/video loading  
- ☁️ **Google Colab** – training environment (Tesla T4 GPU)

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/vehicle-detection-yolov11.git
cd vehicle-detection-yolov11
```

### 2️⃣ Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install required packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python matplotlib seaborn
```

### 4️⃣ Run training or inference

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="custom_data.yaml", epochs=100, imgsz=640)
model("test.jpg").show()
```

---

## 📊 Dataset Overview

- **Source**: [VehicleDetection by Alkan Erturan on Kaggle](https://www.kaggle.com/datasets/alkanerturan/vehicledetection/data)
- **Format**: YOLOv8, adapted for YOLOv11
- **Classes**: `Ambulance`, `Bus`, `Car`, `Motorcycle`, `Truck`
- **Contents**:
  - Labeled images for training, validation, and testing
  - YAML configuration file
  - Evaluation video for real-world performance testing

### 📌 Label Distribution & Placement

![Labels Pairplot](./labels_correlogram.jpg)  
![Labels Summary](./labels.jpg)

---

## ⚙️ Model Information

- **Model**: `YOLOv11n` (nano version)
- **Framework**: PyTorch (via `ultralytics`)
- **Input size**: 640 × 640
- **Loss**: Composite YOLO loss (Objectness + Class + Box)
- **Optimizer**: AdamW
- **Training**: 100 epochs, batch size 16
- **Environment**: Google Colab (Tesla T4 GPU)

---

## 📈 Evaluation Results

| Model      | mAP50 | Precision | Recall |
|------------|-------|-----------|--------|
| YOLOv8n    | 83%   | 86%       | 81%    |
| **YOLOv11n** | **87%**   | **89%**       | **85%**    |

---

### ✅ Confusion Matrix

#### Normalized:
![Confusion Matrix Normalized](./confusion_matrix_normalized.png)

#### Absolute Counts:
![Confusion Matrix Raw](./confusion_matrix.png)

---

## 📉 Performance Curves

| Curve Type            | Image                                 |
|-----------------------|----------------------------------------|
| Recall vs Confidence  | ![Recall](./BoxR_curve.png)            |
| Precision vs Confidence | ![Precision](./BoxP_curve.png)       |
| Precision vs Recall   | ![PR Curve](./BoxPR_curve.png)         |
| F1 vs Confidence      | ![F1 Score](./BoxF1_curve.png)         |

---

## 🚀 Features

- ✅ Real-time vehicle detection on images and video
- ✅ High-performance YOLOv11 with improved accuracy
- ✅ Custom dataset support (5 vehicle classes)
- ✅ Confusion matrix and performance curves
- ✅ Easy integration with PyTorch and Ultralytics

---

## 🧪 Example Usage (in Python)

```python
from ultralytics import YOLO

# Load a YOLOv11 model (nano)
model = YOLO("yolo11n.pt")

# Train on custom dataset
model.train(data="custom_data.yaml", epochs=100, imgsz=640)

# Inference
results = model("test.jpg")
results.show()
```

---

## 📂 File Structure

```
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── video.mp4
├── runs/
│   └── detect/
├── models/
├── results.csv
├── *.png / *.jpg  ← graphs and analysis images
```

---

## 🧠 Future Work

- [ ] Apply transfer learning on `yolo11s.pt` or `yolo11m.pt`
- [ ] Expand dataset with nighttime/weather conditions
- [ ] Deploy model on NVIDIA Jetson Nano or Coral TPU
- [ ] Integrate object tracking (e.g. DeepSORT, ByteTrack)

---

## 📚 Citation

If you use this work, please cite the original dataset:

> Alkan Erturan, *Vehicle Detection Dataset*, [Kaggle](https://www.kaggle.com/datasets/alkanerturan/vehicledetection)

---

## 🌐 Related Links

- 🔗 [YOLOv11 Official Documentation](https://docs.ultralytics.com/models/yolo11/)
- 🔗 [YOLOv8 Docs (legacy format)](https://docs.ultralytics.com/)
- 📘 [YOLOv4 Paper (arXiv)](https://arxiv.org/abs/2004.10934)
