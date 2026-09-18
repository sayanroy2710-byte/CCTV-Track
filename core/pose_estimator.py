"""
Pose Estimation & Structural Geometry Extraction Module.
Uses YOLO11-Pose on CUDA to extract 17 COCO body keypoints,
compute ground-contact points from ankles, and derive scale-invariant
body proportions for multi-cue person tracking.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import config

class PoseDetection:
    def __init__(self, keypoints: np.ndarray, bbox: np.ndarray, conf: float):
        """
        keypoints: (17, 3) array [x, y, conf]
        bbox: (4,) array [x1, y1, x2, y2]
        conf: detection confidence
        """
        self.keypoints = keypoints
        self.bbox = bbox
        self.conf = conf
        self.ground_point = self._compute_ground_point()
        self.structural_vector = self._compute_structural_vector()

    def _compute_ground_point(self) -> Tuple[int, int]:
        """
        Calculates ground plane contact point:
        Uses ankle keypoints if visible (conf >= 0.35),
        otherwise gracefully falls back to bounding-box bottom center.
        """
        x1, y1, x2, y2 = self.bbox
        kp = self.keypoints
        ank_l = kp[15]  # Left Ankle: [x, y, conf]
        ank_r = kp[16]  # Right Ankle: [x, y, conf]
        
        thresh = getattr(config, "POSE_KEYPOINT_CONF_THRESHOLD", 0.35)
        l_vis = ank_l[2] >= thresh
        r_vis = ank_r[2] >= thresh
        
        if l_vis and r_vis:
            # Both ankles visible: contact point is midpoint at lowest Y (closest to floor)
            gx = int((ank_l[0] + ank_r[0]) / 2)
            gy = int(max(ank_l[1], ank_r[1]))
            return (gx, gy)
        elif l_vis:
            return (int(ank_l[0]), int(ank_l[1]))
        elif r_vis:
            return (int(ank_r[0]), int(ank_r[1]))
        else:
            # Fallback (e.g. lower body occluded by counter/desk)
            return (int((x1 + x2) / 2), int(y2))

    def _compute_structural_vector(self) -> np.ndarray:
        """
        Computes scale-invariant body proportions:
        1. Shoulder-to-Torso ratio (shoulder width / torso length)
        2. Hip-to-Shoulder ratio (hip width / shoulder width)
        3. Head-to-Torso ratio (nose to mid-shoulder / torso length)
        4. Torso-to-Leg ratio (torso length / leg length, if leg visible)
        Returns a normalized 4-D feature vector.
        """
        kp = self.keypoints
        thresh = getattr(config, "POSE_KEYPOINT_CONF_THRESHOLD", 0.30)
        
        nose = kp[0]
        sh_l, sh_r = kp[5], kp[6]
        hip_l, hip_r = kp[11], kp[12]
        ank_l, ank_r = kp[15], kp[16]
        
        # 1. Shoulder width
        if sh_l[2] >= thresh and sh_r[2] >= thresh:
            w_sh = float(np.linalg.norm(sh_l[:2] - sh_r[:2]))
        else:
            w_sh = 0.0
            
        # 2. Hip width
        if hip_l[2] >= thresh and hip_r[2] >= thresh:
            w_hip = float(np.linalg.norm(hip_l[:2] - hip_r[:2]))
        else:
            w_hip = 0.0
            
        # 3. Torso length
        if (sh_l[2] >= thresh or sh_r[2] >= thresh) and (hip_l[2] >= thresh or hip_r[2] >= thresh):
            mid_sh = (sh_l[:2] + sh_r[:2]) / 2 if (sh_l[2] >= thresh and sh_r[2] >= thresh) else (sh_l[:2] if sh_l[2] >= thresh else sh_r[:2])
            mid_hip = (hip_l[:2] + hip_r[:2]) / 2 if (hip_l[2] >= thresh and hip_r[2] >= thresh) else (hip_l[:2] if hip_l[2] >= thresh else hip_r[:2])
            l_torso = float(np.linalg.norm(mid_sh - mid_hip))
        else:
            l_torso = 0.0
            
        # 4. Head length
        if nose[2] >= thresh and l_torso > 0:
            mid_sh = (sh_l[:2] + sh_r[:2]) / 2
            l_head = float(np.linalg.norm(nose[:2] - mid_sh))
        else:
            l_head = 0.0
            
        # 5. Leg length
        if (ank_l[2] >= thresh or ank_r[2] >= thresh) and l_torso > 0:
            mid_hip = (hip_l[:2] + hip_r[:2]) / 2
            mid_ank = (ank_l[:2] + ank_r[:2]) / 2 if (ank_l[2] >= thresh and ank_r[2] >= thresh) else (ank_l[:2] if ank_l[2] >= thresh else ank_r[:2])
            l_leg = float(np.linalg.norm(mid_hip - mid_ank))
        else:
            l_leg = 0.0
            
        # Invariant ratios (defaults around human population averages if occluded)
        r_sh_torso = (w_sh / (l_torso + 1e-4)) if (w_sh > 0 and l_torso > 0) else 0.75
        r_hip_sh = (w_hip / (w_sh + 1e-4)) if (w_hip > 0 and w_sh > 0) else 0.65
        r_head_torso = (l_head / (l_torso + 1e-4)) if (l_head > 0 and l_torso > 0) else 0.35
        r_torso_leg = (l_torso / (l_leg + 1e-4)) if (l_leg > 0 and l_torso > 0) else 0.90
        
        # Clamp ratios to plausible human ranges to avoid noise outliers
        r_sh_torso = np.clip(r_sh_torso, 0.40, 1.30)
        r_hip_sh = np.clip(r_hip_sh, 0.40, 1.10)
        r_head_torso = np.clip(r_head_torso, 0.15, 0.65)
        r_torso_leg = np.clip(r_torso_leg, 0.40, 1.60)
        
        vec = np.array([r_sh_torso, r_hip_sh, r_head_torso, r_torso_leg], dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm


class PoseEstimator:
    """
    Manages the YOLO-Pose model on CUDA for multi-person keypoint extraction.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        model_name = model_path or getattr(config, "POSE_MODEL_PATH", "yolo11n-pose.pt")
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def extract_poses(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Runs single-pass pose estimation over frame.
        """
        if frame is None or frame.size == 0:
            return []
            
        results = self.model(frame, verbose=False)
        if not results:
            return []
            
        r = results[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return []
            
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()
        
        detections = []
        for b, c, kp in zip(boxes, confs, kps):
            detections.append(PoseDetection(keypoints=kp, bbox=b, conf=float(c)))
            
        return detections

    @staticmethod
    def match_pose_to_bbox(target_bbox: np.ndarray, pose_detections: List[PoseDetection]) -> Optional[PoseDetection]:
        """
        Finds the pose detection that has the maximum IoU with target_bbox.
        """
        if not pose_detections:
            return None
            
        tx1, ty1, tx2, ty2 = target_bbox
        ta = max(0, tx2 - tx1) * max(0, ty2 - ty1)
        if ta <= 0:
            return None
            
        best_match = None
        best_iou = 0.20  # Minimum IoU threshold to link pose
        
        for p in pose_detections:
            px1, py1, px2, py2 = p.bbox
            pa = max(0, px2 - px1) * max(0, py2 - py1)
            inter = max(0, min(tx2, px2) - max(tx1, px1)) * max(0, min(ty2, py2) - max(ty1, py1))
            if inter > 0:
                iou = inter / (ta + pa - inter)
                if iou > best_iou:
                    best_iou = iou
                    best_match = p
                    
        return best_match
