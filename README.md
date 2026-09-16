# 🏬 Multi-Camera CCTV People Tracking & Re-Identification System

An enterprise-ready AI surveillance solution built with **YOLO11** and **Spatio-Temporal Deep Re-Identification (ReID)**. The system accurately tracks visitors across multiple CCTV cameras in public spaces, shopping malls, retail stores, and campuses.

It automatically records **exact entry and exit timestamps**, calculates **dwell time**, generates a persistent **Global Person ID** (e.g., `Person-001`), renders an integrated **live side analytics dashboard HUD**, persists journeys into an **SQLite database**, and exports the synchronized processed video to disk.

---

## ⚡ Quick Start (1-Click Run)

No complex setup needed. Use the pre-configured Windows batch files:

### 1. Start Live CCTV Tracking (With Graphical Video Selector)
Double-click **`run_live_tracking.bat`** *(or `select_video_and_track.bat`)*  
*(or run in terminal:* `.\C_Yolo\Scripts\python.exe main.py`*)*

- **Interactive GUI Dialog**: Automatically prompts you to browse and select any video file (`.mp4`, `.avi`, `.mkv`), pick multiple videos for a multi-camera grid, or click 1 button to use the default sample video!
- **Real-Time Live HUD**: Once selected, opens the high-resolution tracking window with synchronized live analytics.
- **Saves Output**: The processed video with bounding boxes and live HUD is automatically saved to `output_videos/`.
- Press **`q`** or **`ESC`** on the video window at any time to quit and finalize the recording.

### 2. Open Web Administrator Portal
Double-click **`run_web_dashboard.bat`**  
*(or run in terminal:* `.\C_Yolo\Scripts\streamlit.exe run web_dashboard.py`*)*

- Opens the Streamlit web app in your browser at `http://localhost:8501`.
- View visitor statistics, search any visitor by ID, inspect visual ReID crop thumbnails, and download CSV/Excel journey logs.

---

## 📖 How to Run for Any Video or Setup

The system is fully generalized and can process any video file, multi-camera feeds, USB webcams, or IP network streams:

### 1. Track Default CCTV Video
```bash
.\C_Yolo\Scripts\python.exe main.py
```

### 2. Track Your Own Video File
Provide the path to any `.mp4`, `.avi`, or `.mkv` file:
```bash
.\C_Yolo\Scripts\python.exe main.py --sources "C:\path\to\your_video.mp4"
```

### 3. Track Multiple Camera Feeds Simultaneously
Provide multiple video files separated by space. The system creates a synchronized multi-camera grid:
```bash
.\C_Yolo\Scripts\python.exe main.py --sources "video/entrance.mp4" "video/hallway.mp4" "video/exit.mp4"
```

### 4. Multi-Camera Simulation Mode (Single Video Demo)
Simulates two distinct camera viewpoints (`Cam-01: Entrance` and `Cam-02: Corridor`) from one video source:
```bash
.\C_Yolo\Scripts\python.exe main.py --demo
```

### 5. Live USB Webcams or RTSP Security Cameras
Use camera indices (e.g., `0`, `1`) or RTSP network URLs:
```bash
# Webcam:
.\C_Yolo\Scripts\python.exe main.py --sources 0

# RTSP IP Camera Stream:
.\C_Yolo\Scripts\python.exe main.py --sources "rtsp://admin:password@192.168.1.50:554/stream1"
```

### 6. Background / Headless Mode (Fast Batch Processing)
Processes video without opening a display window (ideal for servers or maximum FPS):
```bash
.\C_Yolo\Scripts\python.exe main.py --headless --output "output_videos/my_tracked_result.mp4"
```

### 7. Continuous Looping
Loop the video continuously for kiosk displays or presentations:
```bash
.\C_Yolo\Scripts\python.exe main.py --loop
```

---

## 🧠 Key Features Explained

| Feature | Description |
| :--- | :--- |
| **1. Entry Time Logging** | When a person first appears on any entrance camera, their exact arrival timestamp is recorded into SQLite. |
| **2. Spatio-Temporal Deep ReID** | Combines deep neural network feature embeddings (MobileNetV3 GPU) with 3-Zone CIELAB clothing color distribution. People maintain their unique ID (e.g., `Person-001`) even if occluded, turned away, or moving across cameras. |
| **3. Multi-Exemplar Gallery** | Stores up to 5 appearance viewpoints per person. When a visitor returns or walks past again, they match their existing ID instead of creating false new tracks. |
| **4. Spatial Aspect-Ratio Filtering** | Rejects square non-human objects and shelf reflections (`h/w >= 1.15`, `min_height >= 40px`) while detecting all genuine pedestrians (both foreground and background). |
| **5. Exit & Dwell Time Calculation** | Measures the exact duration a person spends inside the premises (`Dwell: 0m 45s`) and marks them as `EXITED` upon leaving. |
| **6. Real-Time Side Dashboard HUD** | Built directly into the OpenCV display: displays active count, total footfall, exited count, camera occupancy, active visitor table, and real-time activity log. |
| **7. Database & Snapshot Audit Trail** | Every visitor's entry, exit, camera hops, and best cropped photos are saved automatically to `data/`. |

---

## 📁 Project Directory Structure

```
CCTV People Tracking - Martian/
├── config.py                 # Central configuration: thresholds, model paths, camera setup
├── main.py                   # Main CLI executable & real-time tracking engine
├── web_dashboard.py          # Streamlit web administrator portal
├── custom_botsort.yaml       # Optimized tracker settings (BoT-SORT)
├── run_live_tracking.bat     # 1-Click launcher for real-time tracking
├── run_web_dashboard.bat     # 1-Click launcher for the web portal
│
├── core/
│   ├── tracker.py            # Multi-camera detection, filtering, and tracking orchestrator
│   ├── reid.py               # Deep ReID feature extraction & multi-exemplar memory bank
│   └── journey_manager.py    # Entry/Exit logging, dwell time tracker & SQLite manager
│
├── ui/
│   └── dashboard.py          # Live side HUD renderer (KPI cards, active list, event feed)
│
├── data/
│   ├── cctv_mall_tracking.db # SQLite database storing all visitor journeys & camera events
│   └── person_crops/         # Cropped image thumbnails for every recognized person
│
├── output_videos/            # Recorded processed video files (.mp4)
└── video/
    └── vid1.mp4              # Sample surveillance footage
```

---

## ⚙️ Configuration & Tuning (`config.py`)

You can easily adjust settings in `config.py` without modifying any code:

```python
# 1. Choose Detection Model
YOLO_MODEL_PATH = "yolo11m.pt"        # Options: yolo11n.pt (fastest), yolo11m.pt (balanced), yolo11l.pt (highest accuracy)

# 2. Detection Sensitivity
CONFIDENCE_THRESHOLD = 0.32          # Lower (e.g., 0.25) for distant people; higher (e.g., 0.45) for busy scenes
MIN_PERSON_HEIGHT = 40               # Discards detections smaller than 40 pixels
MIN_PERSON_ASPECT_RATIO = 1.15       # Enforces human standing ratio to reject square reflections/shelves

# 3. Re-Identification (ReID) Sensitivity
REID_SIMILARITY_THRESHOLD = 0.70     # Cosine similarity threshold for matching the same person
MAX_EXEMPLARS_PER_PERSON = 5         # Number of viewpoint photos kept in memory per person

# 4. Exit Timeout
EXIT_TIMEOUT_SECONDS = 15.0          # Seconds without detection before marking a person as EXITED

# 5. Camera Topology
DEFAULT_CAMERAS = {
    "Cam-01": {"name": "Mall Main Entrance", "is_entrance": True, "is_exit": False},
    "Cam-02": {"name": "Central Retail Hallway", "is_entrance": False, "is_exit": False},
    "Cam-03": {"name": "West Exit Gate", "is_entrance": False, "is_exit": True},
}
```

---

## 📊 Where Data is Saved

1. **Processed Tracking Videos**:  
   Saved to `output_videos/` (e.g., `tracked_cctv_final.mp4`). Contains the full surveillance video with bounding boxes, global IDs, and the real-time side dashboard.
2. **Visitor Database**:  
   Saved to `data/cctv_mall_tracking.db`. You can view it through the web dashboard or query with any SQLite viewer:
   - `visitors`: Master table containing `person_id`, `entry_time`, `exit_time`, `dwell_seconds`, `status`, and thumbnail paths.
   - `camera_events`: Log of each time a person enters a specific camera view.
3. **Visitor Photos**:  
   Cropped high-quality image thumbnails are saved to `data/person_crops/Person-XXX.jpg`.

---

## ❓ Frequently Asked Questions (FAQ)

### How do I stop the tracking window?
Click on the video window and press the **`q`** key or **`ESC`** key. The tracker will cleanly close the video, flush all pending journeys to the database, and save the MP4 file.

### How do I reset the tracking data?
Simply delete the contents of the `data/` folder (or delete `data/cctv_mall_tracking.db` and the contents of `data/person_crops/`). Fresh files will be recreated automatically on the next run.

### Can I run this on a computer without an NVIDIA GPU?
Yes. The system automatically detects whether CUDA is available. If an NVIDIA GPU is present, it accelerates YOLO11 and ReID with PyTorch CUDA; otherwise, it falls back to multi-threaded CPU execution.

---

## 📜 License & Credits
Built with **Ultralytics YOLO11**, **PyTorch**, **OpenCV**, and **Streamlit**.