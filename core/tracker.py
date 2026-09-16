"""
Universal Multi-Camera Tracking Pipeline using YOLO11 and Spatio-Temporal ReID.
Orchestrates person detection, batched tracking, aspect ratio validation,
and visual HUD overlays.
"""
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from ultralytics import YOLO

import config
from core.reid import DeepReIDExtractor, ReIDMemoryBank
from core.journey_manager import JourneyManager

COLORS = [
    (255, 105, 65),   # Royal Blue
    (46, 204, 113),   # Emerald Green
    (241, 196, 15),   # Sun Yellow
    (155, 89, 182),   # Amethyst Purple
    (230, 126, 34),   # Carrot Orange
    (26, 188, 156),   # Turquoise
    (231, 76, 60),    # Crimson Red
    (52, 152, 219),   # Dodger Blue
    (255, 128, 171),  # Pink
    (162, 217, 206),  # Mint
    (255, 179, 0),    # Amber
    (0, 230, 118),    # Neon Green
]

def get_color_for_id(global_id: str) -> Tuple[int, int, int]:
    try:
        num = int(global_id.split("-")[-1])
        return COLORS[num % len(COLORS)]
    except Exception:
        return (0, 255, 200)


def deduplicate_boxes(boxes_list):
    """
    Suppresses duplicate / contained boxes on the same individual
    (e.g., separate torso and full-body detections from YOLO).
    """
    if len(boxes_list) <= 1:
        return boxes_list
    boxes_list.sort(key=lambda item: (item[1], (item[2][2]-item[2][0])*(item[2][3]-item[2][1])), reverse=True)
    keep = []
    for lid, conf, xyxy, box in boxes_list:
        x1, y1, x2, y2 = xyxy
        a1 = (x2 - x1) * (y2 - y1)
        dup = False
        for klid, kconf, kxyxy, kbox in keep:
            kx1, ky1, kx2, ky2 = kxyxy
            ka = (kx2 - kx1) * (ky2 - ky1)
            inter = max(0, min(x2, kx2) - max(x1, kx1)) * max(0, min(y2, ky2) - max(y1, ky1))
            if inter > 0:
                ios = inter / min(a1, ka)
                iou = inter / (a1 + ka - inter)
                if ios >= 0.60 or iou >= 0.40:
                    dup = True
                    break
        if not dup:
            keep.append((lid, conf, xyxy, box))
    return keep


class MultiCameraTracker:
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH,
                 tracker_config: str = config.TRACKER_CONFIG_PATH):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker_config = tracker_config
        
        self.reid_extractor = DeepReIDExtractor(device=device)
        self.reid_bank = ReIDMemoryBank(self.reid_extractor)
        self.journey_manager = JourneyManager()
        
        self.trajectories: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
        self.frame_index = 0
        self.local_track_hits: Dict[Tuple[str, int], int] = {}
        self.MIN_CONFIRMED_HITS = 4
        self.live_camera_counts: Dict[str, int] = {}

    def process_camera_batch(self, camera_frames: List[Tuple[str, np.ndarray]],
                             timestamp: datetime) -> List[np.ndarray]:
        if not camera_frames:
            return []
            
        self.frame_index += 1
        frames = [cf[1] for cf in camera_frames]
        
        # Periodic cleanup of track hits
        if self.frame_index % 120 == 0 and len(self.local_track_hits) > 50:
            cached_keys = set(self.reid_bank.local_to_global_cache.keys())
            self.local_track_hits = {k: v for k, v in self.local_track_hits.items() if k in cached_keys}
        
        results = self.model.track(
            source=frames,
            persist=True,
            tracker=self.tracker_config,
            classes=[config.PERSON_CLASS_ID],
            conf=config.CONFIDENCE_THRESHOLD,
            iou=0.45,
            verbose=False
        )
        
        annotated_frames = []
        
        for idx, (camera_id, frame) in enumerate(camera_frames):
            annotated = frame.copy()
            cam_meta = config.DEFAULT_CAMERAS.get(camera_id, {
                "name": camera_id, "is_entrance": False, "is_exit": False
            })
            is_entrance = cam_meta.get("is_entrance", False)
            
            if camera_id not in self.trajectories:
                self.trajectories[camera_id] = {}
                
            r = results[idx]
            frame_assigned_gids: Set[str] = set()
            active_detections_count = 0
            
            # Draw top banner
            banner_color = (32, 36, 44)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), banner_color, -1)
            cam_title = f"{camera_id}: {cam_meta['name']}  (LIVE)"
            cv2.putText(annotated, cam_title, (12, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
            
            if r.boxes is not None and len(r.boxes) > 0:
                raw_boxes = []
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    h_box = y2 - y1
                    w_box = x2 - x1
                    area_box = h_box * w_box
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if h_box < config.MIN_PERSON_HEIGHT or area_box < config.MIN_PERSON_AREA:
                        continue
                        
                    # Geometry & aspect ratio filter: Reject horizontal or square shelf clutter
                    aspect_ratio = h_box / max(1, w_box)
                    min_ratio = getattr(config, "MIN_PERSON_ASPECT_RATIO", 1.15)
                    if aspect_ratio < min_ratio and h_box < 160:
                        continue
                        
                    local_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                    raw_boxes.append((local_id, conf, xyxy, box))
                    
                # Deduplicate overlapping/contained duplicate boxes on same person
                valid_boxes = deduplicate_boxes(raw_boxes)
                # Prioritize established tracks in local cache first
                valid_boxes.sort(key=lambda item: (0 if (camera_id, item[0]) in self.reid_bank.local_to_global_cache else 1, item[0]))
                
                for local_id, conf, xyxy, box in valid_boxes:
                    lid = local_id if local_id >= 0 else None
                    
                    # Tracklet Confirmation: require persistence hits before official ReID registration
                    if lid is not None:
                        is_cached = (camera_id, lid) in self.reid_bank.local_to_global_cache
                        if not is_cached:
                            hits = self.local_track_hits.get((camera_id, lid), 0) + 1
                            self.local_track_hits[(camera_id, lid)] = hits
                            if hits < self.MIN_CONFIRMED_HITS:
                                continue  # Suppress tentative tracklet until confirmed
                    
                    x1, y1, x2, y2 = xyxy
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    
                    cx, cy = int((x1 + x2) / 2), int(y2)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Spatio-Temporal ReID
                    global_id, sim, is_new = self.reid_bank.match_or_register(
                        camera_id=camera_id,
                        local_track_id=lid,
                        crop=person_crop,
                        center=(cx, cy),
                        frame_idx=self.frame_index,
                        timestamp=timestamp,
                        is_entrance=is_entrance,
                        excluded_gids=frame_assigned_gids
                    )
                    
                    frame_assigned_gids.add(global_id)
                    active_detections_count += 1
                    
                    # Update Journey Manager
                    crops = self.reid_bank.get_person_crops(global_id)
                    crop_path = crops[-1] if crops else ""
                    self.journey_manager.update_detection(
                        global_id=global_id,
                        camera_id=camera_id,
                        timestamp=timestamp,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        thumbnail_path=crop_path,
                        is_new=is_new
                    )
                    
                    # Update Trajectory
                    if global_id not in self.trajectories[camera_id]:
                        self.trajectories[camera_id][global_id] = []
                    self.trajectories[camera_id][global_id].append((cx, cy))
                    if len(self.trajectories[camera_id][global_id]) > 30:
                        self.trajectories[camera_id][global_id].pop(0)
                        
                    color = get_color_for_id(global_id)
                    
                    # Draw trajectory trail
                    pts = self.trajectories[camera_id][global_id]
                    for i in range(1, len(pts)):
                        cv2.line(annotated, pts[i - 1], pts[i], color, 2)
                        
                    # Bounding Box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Dwell tag
                    rec = self.journey_manager.active_visitors.get(global_id)
                    dwell_text = rec.dwell_time_str if rec else "00m 00s"
                    
                    font_scale = 0.45 if (x2 - x1) > 50 else 0.35
                    label = f"{global_id} | {dwell_text}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    
                    # Smart tag placement (inside box if near top)
                    if y1 < 35:
                        tag_y1 = y1
                        tag_y2 = y1 + lh + 6
                        text_y = y1 + lh + 2
                    else:
                        tag_y1 = max(0, y1 - lh - 6)
                        tag_y2 = y1
                        text_y = y1 - 3
                        
                    cv2.rectangle(annotated, (x1, tag_y1), (x1 + lw + 8, tag_y2), color, -1)
                    cv2.putText(annotated, label, (x1 + 4, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                                
            # Occupancy badge
            self.live_camera_counts[camera_id] = active_detections_count
            occ_text = f"Occupancy: {active_detections_count}"
            (ow, _), _ = cv2.getTextSize(occ_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(annotated, occ_text, (annotated.shape[1] - ow - 12, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 220, 100), 1, cv2.LINE_AA)
                        
            annotated_frames.append(annotated)
            
        return annotated_frames