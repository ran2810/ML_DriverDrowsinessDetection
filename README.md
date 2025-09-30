# Drowsiness Detection using CNN & Transfer Learning

This project implements a **Drowsiness Detection System** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning with MobileNetV2**.  
It can predict whether a person is **drowsy (eyes closed, yawning)** or **alert (eyes open, not yawning)** in real time using webcam feedback.  

The model supports two approaches:
1. **Single-task classification** → Classifies into 4 categories (`open`, `closed`, `yawn`, `no_yawn`).

---

## 📂 Dataset

We use the [Yawn-Eye Dataset (NEW)](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new).  
The dataset contains labeled images of:
- **Open eyes**
- **Closed eyes**
- **Yawning**
- **Not yawning**

### Dataset Structure
```
data/
│── train/
│   ├── Open/
│   ├── Closed/
│   ├── Yawn/
│   └── No_Yawn/
│── test/
```

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run the Jupyter Notebook:

```bash
jupyter notebook drowsiness-detection.ipynb
```

The notebook supports:
- **Data preprocessing & augmentation**
- **Transfer learning with MobileNetV2**
- **Class balancing using weights**
- **Callbacks** (early stopping, learning rate scheduling, checkpointing)
- **Evaluation** (accuracy, loss curves, confusion matrices)

---


## 🎥 Real-Time Drowsiness Detection

Once trained, you can run the **webcam demo**:

```bash
python webcam_demo.py
```

This will:
- Capture live video
- Detect face, predict **eye state** & **yawning**
- Overlay results on the video feed

Press **`q`** to exit.

---

## 📊 Results

- **Single-task model:** Improved validation accuracy.  
- **Optimized training (with dropout, L2, fine-tuning, augmentation):** Improved generalization.  

---

## 🔧 Troubleshooting

- If validation accuracy is much lower than training → increase augmentation, dropout, or reduce fine-tuned layers.  
- If webcam feed is laggy → reduce input size from `224x224` to `128x128`.  
- If dataset is imbalanced → use `class_weight` during training.  

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt).

---

## 📜 License

MIT License.  
Feel free to use and modify this project for research and learning.  
