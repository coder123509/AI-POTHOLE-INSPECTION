**Pothole Detection System**
A deep learning-based system that automatically detects potholes in images and videos using YOLOv5 and computer vision techniques.


**🚀 Features**
Real-time pothole detection in images and videos
Interactive Streamlit web interface for easy interaction
Advanced analytics and visualization of detection results
Severity classification of detected potholes
Road condition assessment and recommendations

**📋 Project Structure**
pothole-detection/
├── app.py                     # Streamlit web application
├── pothole_detector.py        # Core detection script
├── requirements.txt           # Dependencies
├── models/                    # Trained models
│   └── best.pt                # YOLOv5 trained weights
├── data/                      # Dataset directory (not included in repo)
├── utils/                     # Helper functions
└── README.md                  # Project documentation
🛠️ Setup Instructions
Prerequisites

Python 3.9+ (tested with Python 3.9.21)
CUDA-compatible GPU (optional, for faster inference)

Installation

Clone this repository
bash git clone https://github.com/coder123509/pothole-detection.git
cd pothole-detection

Install dependencies
bashpip install -r requirements.txt

Download the trained model

The pre-trained model is included in the models directory
Alternatively, you can train your own model following the training instructions



**🖥️ Usage
Web Interface**
Run the Streamlit app for a user-friendly interface:
bashstreamlit run app.py
**Features:**

Upload and analyze images
Process videos for pothole detection
View detailed analytics and statistics
Get road condition assessments

Command Line
For batch processing or integration into other systems:
bash# Process an image
python pothole_detector.py --source path/to/image.jpg --output results/output.jpg

# Process a video
python pothole_detector.py --source path/to/video.mp4 --output results/output.mp4

# Use webcam (0 is default camera)
python pothole_detector.py --source 0 --output results/webcam_output.mp4
🧠 Technical Details
Model Architecture

YOLOv5n (nano) for lightweight deployment
Input size: 640x640 pixels
Single class: 'pothole'

**Training Information**

Dataset: Kaggle Pothole Detection dataset (665 images)
Training/validation split: 80/20
Epochs: 10
Batch size: 16
Augmentation: Default YOLOv5 augmentation pipeline

**Performance Metrics**

Precision: 0.777 (77.7%)
Recall: 0.611 (61.1%)
mAP50: 0.724 (72.4%)
mAP50-95: 0.387 (38.7%)

**🔍 Future Improvements**

Mobile application development
GPS integration for pothole mapping
Multi-class detection (pothole severity levels)
Deployment to edge devices for real-time road analysis
Integration with municipal road maintenance systems




Kaggle Pothole Detection Dataset
Ultralytics YOLOv5
Streamlit for the web interface
