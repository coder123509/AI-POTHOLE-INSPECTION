# 🕳️ AI-Pothole-Inspection

A deep learning-based system that automatically detects potholes in uploaded images and videos using **YOLOv5** and computer vision techniques. This project is powered by a lightweight yet powerful model and presented through an intuitive **Streamlit** interface.

---

## 🚀 Features

- 📷 Upload and analyze **images** for potholes  
- 🎥 Process **videos** for pothole detection  
- 📊 Visual analytics and detection statistics  
- ⚠️ Severity classification based on pothole size  
- 🚣️ Road condition assessment and smart recommendations  

---

## 📁 Project Structure

```
AI-POTHOLE-INSPECTION/
                  
├── pothole_detector.py     # Core detection script+streamlit-web-interface
├── requirements.txt        # Dependencies
├── models/                 # Trained YOLOv5 model weights
│   └── best.pt
├── utils/                  # Helper functions
├── results/                # Output images/videos (generated)
└── README.md               # Project documentation
```

---

## 💪 Setup Instructions

### ✅ Prerequisites

- Python 3.9+ (tested with Python 3.9.21)  
- CUDA-compatible GPU (optional, for faster inference)

### 📦 Installation

1. Clone the repository  
```bash
git clone https://github.com/coder123509/AI-POTHOLE-INSPECTION.git
cd AI-POTHOLE-INSPECTION
```

2. Install the required dependencies  
```bash
pip install -r requirements.txt
```

3. (Optional) Create and activate a virtual environment  
```bash
conda create -n pothole-env python=3.9
conda activate pothole-env
```

---

## 🧠 Model Details

- **Model**: YOLOv5n (nano) – lightweight and fast  
- **Input size**: 640×640 pixels  
- **Classes**: Single class - `pothole`

### 📊 Training Info

- **Dataset**: [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets)  
- **Train/Validation Split**: 80/20  
- **Epochs**: 10  
- **Batch size**: 16  
- **Augmentations**: YOLOv5 default pipeline

### 📈 Performance Metrics

| Metric      | Score  |
|-------------|--------|
| Precision   | 77.7%  |
| Recall      | 61.1%  |
| mAP@0.5     | 72.4%  |
| mAP@0.5:0.95| 38.7%  |

---

## 💻 Usage

### 🌐 Web Interface (Streamlit)

Launch the app:
```bash
streamlit run pothole_detector.py
```

#### Web App Features:
- Upload images for pothole detection
- Upload videos for analysis
- Get visual feedback, pothole stats, and road severity level



## 🔍 Future Roadmap

- 📱 Mobile application version  
- 🗑️ GPS-based pothole mapping  
- 📶 Real-time deployment on edge devices  
- 🧠 Multi-class classification for pothole types  
- 🏡 Integration with government road monitoring systems  

---

## 🤝 Built With

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  
- [Streamlit](https://streamlit.io)  
- [OpenCV](https://opencv.org)  
- [Kaggle Pothole Detection Dataset](https://www.kaggle.com)  

---

## 📌 Note

This app currently supports **image and video uploads** only.  
Live webcam support is planned for a future release.

---



## 👨‍💼 Authors

- Abhinav Krishna Rayachoti
  
  
---

