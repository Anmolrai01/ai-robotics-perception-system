# AI Robotics Perception System

An AI-driven real-time robotics perception system developed using YOLOv8 and OpenCV featuring object detection, object tracking, spatial awareness, distance estimation, threat analysis, and autonomous navigation decision logic.

---

# Features

- Real-time object detection using YOLOv8
- Multi-object tracking with unique IDs
- Distance estimation (NEAR / MEDIUM / FAR)
- Spatial awareness using LEFT / CENTER / RIGHT regions
- Autonomous navigation decision logic
- Threat assessment system
- Multi-object scene analysis
- Real-time FPS monitoring
- CSV detection logging
- Voice alert system

---

# Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- pyttsx3

---

# Robotics Logic

The system performs real-time scene understanding and generates navigation decisions such as:

- STOP : OBJECT AHEAD
- TURN LEFT
- TURN RIGHT
- PATH CLEAR

based on object position and estimated distance.

---

# Project Structure

```text
ai-robotics-perception-system/
│
├── main.py
├── requirements.txt
├── detection_log.csv
├── sample1.png
├── sample2.png
└── README.md
```

---

# Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

# Screenshots

## Real-Time Robotics Perception

![Sample 1](sample1.png)

![Sample 2](sample2.png)

---

# Applications

- Robotics perception systems
- Autonomous navigation
- Obstacle detection
- Human-aware robotics
- AI-based surveillance systems

---

# Future Improvements

- ROS integration
- Custom-trained object detection models
- Depth estimation using stereo vision
- Embedded deployment on Jetson Nano / Raspberry Pi
- SLAM integration

---

# Author

Anmol Rai
