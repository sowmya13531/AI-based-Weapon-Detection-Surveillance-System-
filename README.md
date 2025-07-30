## 🔫 AI-based Weapon Detection Surveillance System

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-orange?logo=streamlit)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green?logo=github)](https:/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)

A real-time AI surveillance system built with **YOLOv8** and **Streamlit** to detect weapons (guns, knives, rifles) from webcam, images, and videos.
> 🚨 This project helps automate threat detection for safety monitoring in public and private spaces.

## 📷 Features
- ✅ Real-time webcam detection  
- ✅ Upload and analyze images and videos  
- ✅ Automatic logging of detected weapons with time, confidence, and source  
- ✅ Downloadable detection log (CSV)  
- ✅ Lightweight and easy to deploy

## 🛠️ Tech Stack
- **Python 3.9+**
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- OpenCV
- NumPy, Pandas, Pillow

## 📦 Installation
1. **Clone the repo**

```bash
git clone https://github.com/sowmya13531/AI-based-Weapon-Detection-Surveillance-System-.git
cd AI-based-Weapon-Detection-Surveillance-System-
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the app
- streamlit run weapon_detection_app.py

## 🧠 How It Works
- >Loads a pretrained YOLOv8 model (best(3).pt)
- >Detects weapons based on defined class mappings (gun, knife, etc.)
### Supports 4 modes:
- 🎥 Webcam
- 🖼️ Upload Image
- 🎞️ Upload Video
- 📑 Upload CSV (for visualizing past detections)

- >Logs detection info into a detection_log.csv file
- >Saves annotated frames if detection is successful

## 📁 Folder Structure
bash
- ├── app.py                  # Main Streamlit app
- ├── best.pt                 # Custom YOLOv8 model file
- ├── saved_frames/           # Detected frames will be stored here
- ├── detection_log.csv       # Log file for all detections
- ├── requirements.txt        # Dependencies
- └── README.md               # This file

## 🛡️ Weapon Classes Supported
## Class Detected	Mapped To
->handgun	pistol
->sharp_object	knife
->assault_rifle	rifle
Only detections matching keywords like gun, knife, rifle, etc. are logged.

## 📊 Sample Detection Log

2025-07-30T14:22:10Z	webcam	pistol	0.91	saved_frames/pistol_1.jpg

## 🙋‍♀️ Contributing
- Pull requests are welcome! Feel free to fork the repo and submit improvements.

### ✨ Acknowledgments
Ultralytics YOLOv8
Streamlit

## 🚀 Future Enhancements
- Add real-time alert notifications via SMS or email for detected threats.
- Improve model accuracy with more diverse training data and weapon types.
- Integrate multi-camera support and cloud-based remote monitoring.
