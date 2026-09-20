"""
Universal Configuration for Multi-Camera CCTV People Tracking & Re-Identification.
Optimized for high-precision person tracking with spatio-temporal ReID.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_DIR = BASE_DIR / "output_videos"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "cctv_mall_tracking.db"
CROPS_DIR = DATA_DIR / "person_crops"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Detection & Tracking Settings
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11m.pt")
TRACKER_CONFIG_PATH = str(BASE_DIR / "custom_botsort.yaml")
CONFIDENCE_THRESHOLD = 0.30          # High-sensitivity detection (matches original download)
PERSON_CLASS_ID = 0                  # COCO class 0 is person
MIN_PERSON_HEIGHT = 45               # Minimum pixel height for person
MIN_PERSON_AREA = 1500               # Discard tiny reflection/shelf noise while retaining real people

# Person Re-Identification (ReID) Parameters
REID_BACKBONE = "mobilenet_v3_small"
REID_SIMILARITY_THRESHOLD = 0.68     # High-precision multi-exemplar threshold
REID_FEATURE_ALPHA = 0.85            # Momentum update for feature gallery
MAX_EXEMPLARS_PER_PERSON = 8         # Diverse viewpoint templates learned per person
MAX_GALLERY_IMAGES_PER_PERSON = 8    # Clicked/captured crops stored in gallery per person

# Pose Estimation & Multi-Cue Fusion Parameters
ENABLE_POSE_ESTIMATION = True        # Enable YOLO11-Pose for keypoints & ground anchoring
POSE_MODEL_PATH = str(BASE_DIR / "yolo11n-pose.pt")
POSE_KEYPOINT_CONF_THRESHOLD = 0.35  # Confidence for valid keypoints
FUSION_HIGH_CONF_THRESHOLD = 0.68    # Tier 1 auto-association threshold
FUSION_MEDIUM_CONF_THRESHOLD = 0.52  # Tier 2 tentative association threshold
DRAW_POSE_SKELETON = True            # Render subtle pose keypoint overlay in output video

# Motion Verification / Static Object Rejection (Requirement 5: if still then not a person)
FILTER_STATIC_OBJECTS = True         # Filter out non-human stationary objects (mannequins, posters, cutouts)
MIN_MOTION_DISPLACEMENT_PX = 3.5     # Minimum pixel movement required over observation window
STATIC_OBSERVATION_WINDOW_FRAMES = 30# Evaluation window frames before confirming genuine human motion

# Journey and Dwell Time Parameters
DEFAULT_CAMERAS = {
    "Cam-01": {"name": "Mall Main Entrance / Hallway", "is_entrance": True, "is_exit": False},
    "Cam-02": {"name": "Central Retail Zone", "is_entrance": False, "is_exit": False},
    "Cam-03": {"name": "Exit Corridor", "is_entrance": False, "is_exit": True},
}

# Timeout in seconds without detection before marked as exited
EXIT_TIMEOUT_SECONDS = 15.0

# Display & UI Settings
MAX_DISPLAY_HEIGHT = 720
DASHBOARD_WIDTH = 420
DASHBOARD_BG_COLOR = (24, 28, 36)
DASHBOARD_CARD_BG = (36, 42, 54)
ACCENT_CYAN = (235, 206, 0)
ACCENT_GREEN = (80, 220, 100)
ACCENT_ORANGE = (30, 140, 255)
ACCENT_RED = (60, 60, 240)
TEXT_WHITE = (245, 245, 245)
TEXT_MUTED = (160, 165, 175)

# Video Export Settings
EXPORT_FPS = 24.0
VIDEO_CODEC = "mp4v"