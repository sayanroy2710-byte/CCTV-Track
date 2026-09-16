# 🏬 AI-Based CCTV People Tracking & Dwell-Time Analytics System

An enterprise-ready AI surveillance and video analytics solution built with **YOLO11** and **Spatio-Temporal Deep Person Re-Identification (ReID)**. The system accurately detects, tracks, and analyzes visitors across single or multi-camera CCTV setups in retail malls, commercial stores, transportation hubs, and smart facilities.

---

## 🌟 Key Capabilities
* **Interactive Graphical Launcher**: Easily browse and select any video file (`.mp4`, `.avi`, `.mov`, `.mkv`), configure multi-camera grids, or connect live RTSP/webcam feeds via an intuitive GUI.
* **Persistent Cross-Camera ReID**: Preserves unique individual identities (e.g., `Person-001`) even through severe occlusions, viewpoint changes, and camera handovers using deep neural appearance embeddings (MobileNetV3) combined with 3-zone CIELAB spatial color signatures.
* **Accurate Dwell-Time Analytics**: Automatically logs exact entry timestamps, real-time presence duration, departure events, and total dwell times.
* **Configurable Exit Waiting Timer**: Prevents false departures due to temporary occlusions with an adjustable inactivity buffer (default: 15 seconds).
* **Live Side Analytics Dashboard (HUD)**: Renders real-time Key Performance Indicators (Total Footfall, Active Count, Departures, Camera Occupancy, and Live Event Feeds) directly alongside the surveillance video stream.
* **Historical Web Management Portal**: Includes a browser-based Streamlit dashboard for supervisors to inspect SQL visitor registries, query individual journey timelines, and download CSV/Excel reports.

---

## 📋 System Prerequisites

Before starting, ensure you have:
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
* **Python**: Version **3.10** or **3.11** (Recommended: Python 3.10.11)
* **Git**: Installed and accessible in your command line
* **Hardware**:
  * **GPU (Recommended)**: NVIDIA GPU with CUDA support for real-time 30+ FPS tracking.
  * **CPU (Supported)**: The system automatically falls back to multi-threaded CPU execution if no GPU is detected.

---

## 🚀 Step-by-Step Local Installation Guide

Follow these steps to download and set up the project on your local machine:

### Step 1: Clone the Repository
Open your terminal (PowerShell, Command Prompt, or Bash) and clone the repository:

```bash
git clone https://github.com/sayanroy2710-byte/CCTV-Track.git
cd CCTV-Track
```

---

### Step 2: Create a Python Virtual Environment

It is strongly recommended to use an isolated virtual environment to prevent dependency conflicts.

* **On Windows (PowerShell / CMD):**
  ```powershell
  python -m venv C_Yolo
  .\C_Yolo\Scripts\activate
  ```

* **On Linux / macOS:**
  ```bash
  python3 -m venv C_Yolo
  source C_Yolo/bin/activate
  ```

*(Once activated, your terminal prompt will display `(C_Yolo)`).*

---

### Step 3: Install Required Dependencies

Upgrade `pip` and install all necessary packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **💡 GPU Acceleration Note (Optional but Recommended):**  
> If you have an NVIDIA graphics card, install PyTorch with CUDA support for optimal performance:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Step 4: Model Weights Auto-Download

You **do not** need to search for or manually download model weights!  
On the very first launch, the system automatically downloads the official Ultralytics YOLO11 weights (`yolo11m.pt`) and PyTorch MobileNetV3 feature extractor weights into the root folder.

---

## 🎮 How to Run the System

### Option 1: 1-Click Launchers (Windows Users)

If you are on Windows, you can simply double-click the pre-configured batch scripts in the project directory:

1. **`run_live_tracking.bat`** or **`select_video_and_track.bat`**  
   Opens the **Graphical Video Selector GUI**. Select your video and click **"Start Live CCTV Tracking"**.
2. **`run_web_dashboard.bat`**  
   Opens the browser-based **Streamlit Web Portal** at `http://localhost:8501`.

---

### Option 2: Running via Command Line (CLI)

Ensure your virtual environment is active (`.\C_Yolo\Scripts\activate`), then run any of the following:

#### 1. Interactive Graphical Launcher (Default)
```bash
python main.py
```
This launches the graphical selector dialog where you can browse video files, choose detection models, adjust confidence thresholds, and tune the exit timeout.

#### 2. Process the Included Sample CCTV Video Directly
```bash
python main.py --no-gui
```

#### 3. Process Any Custom Video File
Provide the file path to any `.mp4`, `.avi`, `.mov`, or `.mkv` video:
```bash
python main.py --sources "C:\path\to\your_video.mp4"
```

#### 4. Synchronized Multi-Camera Grid (Multiple Feeds)
Pass multiple video files separated by space to simulate multiple CCTV cameras simultaneously:
```bash
python main.py --sources "video/entrance.mp4" "video/hallway.mp4" "video/exit.mp4"
```

#### 5. Simulated 2-Camera Topology (Single Video Demo)
Splits a single video into two time-offset camera streams (`Cam-01: Entrance` and `Cam-02: Corridor`) to test cross-camera re-identification:
```bash
python main.py --demo
```

#### 6. Live USB Webcams or IP Security Cameras (RTSP)
```bash
# Connect local USB webcam:
python main.py --sources 0

# Connect network IP camera via RTSP stream:
python main.py --sources "rtsp://admin:password@192.168.1.100:554/stream1"
```

#### 7. Headless Mode (Maximum FPS for Batch/Server Ingestion)
Processes video in the background without rendering a window on screen (ideal for servers or maximum export speed):
```bash
python main.py --headless --sources "video/vid1.mp4" --output "output_videos/batch_output.mp4"
```

---

### Option 3: Launching the Web Administrator Portal

To explore real-time metrics, visitor search, dwell-time charts, and visitor crop galleries:

```bash
streamlit run web_dashboard.py
```
*(Automatically opens your default web browser at `http://localhost:8501`)*.

---

## 🖥️ Using the Graphical Video Selector GUI

When `python main.py` runs, it presents a user-friendly launcher dialog:

| Control | Description |
| :--- | :--- |
| **Single Video File** | Select and process a single surveillance camera video. |
| **Multi-Camera Grid** | Add multiple video files to build a synchronized multi-camera view. |
| **Live Webcam / RTSP** | Enter a camera index (e.g. `0`) or RTSP URL for live streaming. |
| **Browse Video...** | Opens the standard file explorer to choose any video file on your computer. |
| **Use Sample CCTV Video** | 1-click preset that immediately loads the built-in sample footage (`vid1.mp4`). |
| **Detection Model** | Choose between `yolo11m.pt` (balanced default), `yolo11n.pt` (ultra-fast nano), or `yolo11l.pt` (high accuracy). |
| **Confidence Gate** | Detection sensitivity threshold (default: `0.28`). |
| **Exit Waiting Timer (s)** | Inactivity buffer duration before marking a visitor as exited (default: `15.0s`). |
| **Simulate 2-Camera Topology** | Checkbox flag to activate multi-camera simulation (`--demo`). |
| **Continuous Video Loop** | Checkbox flag to repeat playback continuously (`--loop`). |
| **Start Live CCTV Tracking** | Green button to launch the live computer vision tracking engine. |
| **Open Web Portal** | Blue button to launch the administrative browser dashboard. |

---

## ⌨️ Runtime Hotkeys & Controls

While the live CCTV tracking window is running:
* Press **`q`** or **`ESC`**: Stops processing cleanly, flushes all active journeys into SQLite, finalizes video recording, and exits.
* **Window Resizing**: The OpenCV window is responsive and can be resized or maximized to fit your screen.

---

## 📂 Project Directory Structure

```
CCTV-Track/
├── config.py                 # Central system parameters, paths, and thresholds
├── main.py                   # Primary application entry point & tracking pipeline
├── web_dashboard.py          # Streamlit administrative portal
├── custom_botsort.yaml       # Tuned BoT-SORT multi-object tracking settings
├── requirements.txt          # Python dependencies
├── run_live_tracking.bat     # Windows 1-click launcher for CCTV tracking
├── run_web_dashboard.bat     # Windows 1-click launcher for Web Portal
├── select_video_and_track.bat# Windows 1-click launcher for Video Selector GUI
│
├── core/
│   ├── tracker.py            # Multi-camera detector, aspect-ratio filter & tracker
│   ├── reid.py               # Deep ReID feature extractor & multi-exemplar gallery
│   └── journey_manager.py    # Dwell-time calculator, state machine & SQLite logger
│
├── ui/
│   ├── dashboard.py          # Real-time side analytics HUD renderer (OpenCV)
│   └── video_selector_gui.py # Tkinter graphical video input launcher
│
├── data/
│   ├── cctv_mall_tracking.db # SQLite database storing visitors, dwell times & events
│   └── person_crops/         # High-resolution image crops of tracked individuals
│
├── output_videos/            # Recorded MP4 outputs with bounding boxes & side HUD
└── video/
    └── vid1.mp4              # Sample surveillance footage for quick testing
```

---

## ⚙️ Configuration & Customization (`config.py`)

You can customize core behavioral thresholds directly in `config.py`:

```python
# Detection & Tracking Model
YOLO_MODEL_PATH = "yolo11m.pt"        # Detection weights: yolo11n.pt, yolo11m.pt, yolo11l.pt
CONFIDENCE_THRESHOLD = 0.28          # Sensitivity threshold (0.15 - 0.60)
MIN_PERSON_HEIGHT = 45               # Minimum pixel height to filter distant noise
MIN_PERSON_ASPECT_RATIO = 1.15       # Minimum height/width ratio to reject square reflections

# Deep Re-Identification (ReID)
REID_SIMILARITY_THRESHOLD = 0.68     # Similarity cutoff for identity matching
MAX_EXEMPLARS_PER_PERSON = 5         # Number of viewpoint appearance crops saved per person

# Temporal Journey Management
EXIT_TIMEOUT_SECONDS = 15.0          # Inactivity buffer before recording visitor departure

# Display Settings
DASHBOARD_WIDTH = 420                # Pixel width of the integrated side analytics HUD
```

---

## ❓ Frequently Asked Questions (FAQ)

#### Q1: Where are the tracking results and videos saved?
* **Processed Videos**: Automatically exported to `output_videos/tracked_run_TIMESTAMP.mp4`.
* **Visitor Database**: Stored in `data/cctv_mall_tracking.db` (compatible with any SQLite browser or DB viewer).
* **Visitor Thumbnails**: Stored in `data/person_crops/`.

#### Q2: How do I clear past tracking history?
Delete `data/cctv_mall_tracking.db` and the image files inside `data/person_crops/`. Fresh database tables will be initialized automatically on the next run.

#### Q3: Can I run this on a laptop without a dedicated GPU?
Yes! The system automatically detects hardware capabilities. On CPUs, YOLO11 uses multi-threaded inference, and the MobileNetV3 ReID backbone runs efficiently in real-time.

---

## 📜 License & Acknowledgments
* Detection: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
* Deep Learning: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
* Computer Vision: [OpenCV](https://opencv.org/)
* Web Dashboard: [Streamlit](https://streamlit.io/)
