#  Suspicious Activity Detection from CCTV Footage

> An intelligent surveillance system that automatically detects suspicious human behavior in CCTV footage using Deep Learning and Computer Vision.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Demo](#demo)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Advantages](#advantages)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Team](#team)
- [Conclusion](#conclusion)

---

##  Overview

In today's world, CCTV cameras are installed everywhere — malls, banks, streets, and public spaces. However, manually monitoring hundreds of live camera feeds is practically impossible. This project solves that problem by automating the detection of suspicious activity using a trained deep learning model.

The application reads video footage, extracts individual frames, and classifies each frame as either **suspicious** or **normal** with a confidence probability. If more than 10 consecutive frames are flagged as suspicious with over 80% confidence, the system raises an alert.

---

##  Problem Statement

State CCTV Control Rooms receive feeds from thousands of cameras across cities. It is physically impossible for human operators to monitor all feeds in real time. Delayed detection of suspicious activity can lead to crimes going unnoticed until it is too late.

**Key challenges with existing systems:**
- Low accuracy in automated detection
- Low efficiency under real-world conditions
- High dependency on manual monitoring
- No real-time alerting mechanism

---

##  Solution

This application automates the process of suspicious activity detection by:

1. Accepting any CCTV video file as input
2. Automatically extracting up to 500 frames from the video
3. Running each frame through a trained ResNet50 deep learning model
4. Flagging frames where suspicious activity is detected with high confidence
5. Displaying the results in a clean, easy-to-read GUI

---

##  Features

- **Video Upload** — Supports `.mp4` and `.webm` video formats
- **Automatic Frame Extraction** — Extracts up to 500 frames from any uploaded video
- **AI-Powered Detection** — Uses a trained ResNet50 CNN model for classification
- **Real-time Logging** — Displays frame-by-frame processing status live in the GUI
- **Confidence Scoring** — Shows the probability percentage for each detection
- **Consecutive Frame Logic** — Only triggers an alert after 10+ consecutive suspicious frames (reduces false positives)
- **Dual Text Panel GUI** — Left panel shows frame extraction logs, right panel shows detection results
- **No Internet Required** — Fully offline once installed

---

##  Demo

### Normal Activity Detected
When the uploaded video contains normal behavior, the right panel displays:
```
No suspicious activity found in given footage
```

### Suspicious Activity Detected
When suspicious activity is found, the right panel displays:
```
frames/frame117.jpg is predicted as suspicious with probability: 92.45

frames/frame118.jpg is predicted as suspicious with probability: 89.12
```

---

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| TensorFlow 2.21.0 | Deep learning backend |
| Keras 3.13.2 | Model architecture and weight loading |
| ResNet50 (CNN) | Pre-trained base model for classification |
| OpenCV 4.13 | Video processing and frame extraction |
| Tkinter | Desktop GUI framework |
| NumPy 2.4.4 | Array operations and preprocessing |
| imutils | Image path utilities |
| Pillow 12.1.1 | Image loading and processing |
| h5py | Loading `.h5` model weight files |

### Why ResNet50?
ResNet50 (Residual Network with 50 layers) is a powerful Convolutional Neural Network architecture that uses skip connections to avoid the vanishing gradient problem. It was pre-trained on ImageNet and fine-tuned on our dataset to classify frames as suspicious or normal with high accuracy.

---

##  System Requirements

### Hardware
| Component | Minimum |
|---|---|
| Processor | Intel Core i5 or equivalent |
| RAM | 8 GB |
| Storage | 10 GB free space |
| OS | Windows 10/11 (64-bit) |

### Software
| Component | Version |
|---|---|
| Python | 3.11 |
| pip | Latest |

---

##  Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/Rasmitha06/Suspicious-Activity-Detection.git
cd Suspicious-Activity-Detection
```

### Step 2 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Verify the required files are present
Make sure these files exist in the project folder:
```
 SuspiciousDetection.py
 model.h5
 model_class.json
 videos/ (folder with sample videos)
```

---

##  How to Run

```bash
python SuspiciousDetection.py
```

The GUI window will open automatically.

---

##  How It Works

```
Step 1 — Click "Upload CCTV Footage"
         Select any .mp4 or .webm video file from the videos/ folder

Step 2 — Click "Generate Frames"
         The app automatically creates a frames/ folder
         Extracts up to 500 frames from the video
         Saves them as frame0.jpg, frame1.jpg ... frame499.jpg
         Left panel shows each frame being saved in real time

Step 3 — Click "Detect Suspicious Activity Frame"
         Each frame is preprocessed and passed through the ResNet50 model
         If probability > 80% → suspicious frame counter increases
         If 10+ consecutive suspicious frames → alert is logged
         Right panel displays all flagged frames with confidence scores
```

### Detection Logic
```python
if probability > 80%:
    consecutive_count += 1
else:
    consecutive_count = 0

if consecutive_count > 10:
    → Log as suspicious activity detected
```

This consecutive-frame logic ensures the system does not trigger false alarms on a single ambiguous frame.

---

## 📁 Project Structure

```
Suspicious-Activity-Detection/
│
├── SuspiciousDetection.py     # Main application code
├── model.h5                   # Trained ResNet50 model weights (94 MB)
├── model_class.json           # Class label mapping {"0": "normal", "1": "suspicious"}
├── requirements.txt           # All Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
└── videos/                    # Sample CCTV footage for testing
    ├── normal.mp4             # Normal activity video
    ├── normal1.mp4            # Normal activity video
    ├── video2.webm            # Suspicious activity video
    └── video3.mp4             # Suspicious activity video
```

> **Note:** The `frames/` folder is auto-generated when you click "Generate Frames" and does not need to be created manually.

---

##  Model Details

| Property | Value |
|---|---|
| Base Architecture | ResNet50 |
| Input Size | 224 × 224 × 3 (RGB) |
| Output Classes | 2 (normal, suspicious) |
| Final Layer | Softmax activation |
| Preprocessing | ResNet50 standard normalization |
| Training Framework | TensorFlow / Keras 2.2.4 |
| Model File Format | HDF5 (.h5) — weights only |
| Model Size | ~94 MB |

### Training Details
- The model was trained on images of people exhibiting suspicious behavior (face covering, shoplifting-like poses) versus normal behavior
- Custom classification head added on top of ResNet50 base: `GlobalAveragePooling2D → Dense(2) → Softmax`
- Weights loaded using `by_name=True` to match the original ImageAI v2.x layer naming convention

---

## Advantages

| Advantage | Description |
|---|---|
| **High Accuracy** | ResNet50 CNN provides reliable classification results |
| **High Efficiency** | Processes frames quickly without a GPU |
| **Fully Automated** | No manual frame-by-frame review needed |
| **No Existing Database Required** | Works purely on visual analysis, not pre-stored criminal data |
| **Affordable** | Uses open-source tools only, no paid services |
| **Offline** | No internet connection required after installation |
| **Configurable** | Probability threshold and frame count easily adjustable in code |

---

##  Limitations

- Currently detects suspicious activity based on **face covering and similar behaviors** — the model is trained on specific patterns
- Works on **pre-recorded video** only — live CCTV stream integration is not yet implemented
- Maximum **500 frames** extracted per video
- Detection accuracy depends on video quality and lighting conditions

---

##  Future Scope

-  **Live stream support** — Connect directly to IP cameras or RTSP streams
-  **Multi-person tracking** — Detect suspicious activity from multiple people simultaneously
-  **Alert notifications** — Send email or SMS alerts when suspicious activity is detected
-  **Database logging** — Store detection results with timestamps in a database
-  **Web dashboard** — Replace the Tkinter GUI with a browser-based interface
-  **More activity types** — Expand the model to detect fighting, theft, vandalism, etc.

---

##  Team

This project was developed as part of an Innovative Product Development submission at:

**Malla Reddy Engineering College for Women**
Department of Information Technology
Maisammaguda, Secunderabad, Telangana

| Name | Roll Number |
|---|---|
| CH. Rasmitha | 20RH1A0549 |
| CH. Sindhu | 20RH1A0550 |
| D. Satvika | 20RH1A0557 |

**Guide:** Dr. Jayarajan, Assistant Professor

---

##  Conclusion

The Suspicious Activity Detection system demonstrates how deep learning can be applied to real-world surveillance challenges. By automating the analysis of CCTV footage, this system significantly reduces the human effort required to monitor camera feeds while maintaining high detection accuracy.

The system's consecutive-frame detection logic minimizes false positives, and its straightforward GUI makes it accessible to non-technical users. It is affordable, offline-capable, and does not rely on any external database, making it a practical solution for real-world deployment.

---

##  References

1. LOVECEK, T. VELAS, A. DUROVEC, M. (2015). *Bezpecnostne systemy: poplachove systemy*. EDIS. ISBN: 978-80-554-1144-6
2. CAPUTO, C. A. (2014). *Digital Video Surveillance and Security*. Second edition. Elsevier. ISBN: 978-0-12-420042-5
3. NILSON, F. (2009). *Intelligent Network Video: Understanding Modern Video Surveillance Systems*. CRC Press. ISBN: 978-1-4200-6156
4. STN EN 62676-4, 2015. Video surveillance systems for use in security applications.
5. TensorFlow Documentation — https://www.tensorflow.org
6. ImageAI Documentation — https://imageai.readthedocs.io

---

<p align="center">
  Made with dedication by the Crime Detector Team · Malla Reddy Engineering College for Women · 2023
</p>
