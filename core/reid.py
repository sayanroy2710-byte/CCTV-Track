"""
Universal Spatio-Temporal Person Re-Identification (ReID) Engine.
Unifies visual appearance embeddings with spatial-temporal continuity
priors to eliminate tracklet fragmentation, prevent false identity switches,
and accurately track individuals across any CCTV video.
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import torchvision.models as models
import torchvision.transforms as transforms
import config

class DeepReIDExtractor:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model = models.mobilenet_v3_small(weights=weights)
        
        self.backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten()
        ).to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _extract_spatial_color_signature(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return np.zeros(216, dtype=np.float32)
        try:
            resized = cv2.resize(crop, (64, 128))
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            h = lab.shape[0]
            
            z1 = lab[:int(h * 0.35), :]
            z2 = lab[int(h * 0.35):int(h * 0.70), :]
            z3 = lab[int(h * 0.70):, :]
            
            h1 = cv2.calcHist([z1], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256]).flatten()
            h2 = cv2.calcHist([z2], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256]).flatten()
            h3 = cv2.calcHist([z3], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256]).flatten()
            
            hist = np.concatenate([
                h1 / (np.linalg.norm(h1) + 1e-6),
                h2 / (np.linalg.norm(h2) + 1e-6),
                h3 / (np.linalg.norm(h3) + 1e-6)
            ])
            return (hist / (np.linalg.norm(hist) + 1e-6)).astype(np.float32)
        except Exception:
            return np.zeros(216, dtype=np.float32)

    @torch.no_grad()
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 576 + 216), dtype=np.float32)
            
        tensors = []
        color_feats = []
        
        for crop in crops:
            if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_crop)
                tensors.append(tensor)
                color_feats.append(self._extract_spatial_color_signature(crop))
            except Exception:
                continue
                
        if not tensors:
            return np.empty((0, 576 + 216), dtype=np.float32)
            
        batch_tensors = torch.stack(tensors).to(self.device)
        deep_embeddings = self.backbone(batch_tensors).cpu().numpy()
        
        d_norms = np.linalg.norm(deep_embeddings, axis=1, keepdims=True) + 1e-6
        deep_embeddings = deep_embeddings / d_norms
        
        color_embeddings = np.stack(color_feats)
        c_norms = np.linalg.norm(color_embeddings, axis=1, keepdims=True) + 1e-6
        color_embeddings = color_embeddings / c_norms
        
        fused = np.hstack([deep_embeddings * 0.45, color_embeddings * 0.55])
        f_norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-6
        fused = fused / f_norms
        
        return fused.astype(np.float32)


class PersonMemory:
    def __init__(self, global_id: str, initial_embedding: np.ndarray, initial_crop: np.ndarray,
                 camera_id: str, center: Tuple[int, int], frame_idx: int, timestamp: datetime):
        self.global_id = global_id
        self.exemplars: List[np.ndarray] = [initial_embedding.copy()]
        self.crops: List[str] = []
        self.total_detections = 1
        
        self.last_camera = camera_id
        self.last_center = center
        self.last_frame = frame_idx
        self.last_timestamp = timestamp
        self.save_crop(initial_crop)

    def match_score(self, candidate_embedding: np.ndarray, camera_id: str,
                    center: Tuple[int, int], frame_idx: int) -> Tuple[float, float]:
        sims = [float(np.dot(candidate_embedding, ex)) for ex in self.exemplars]
        raw_sim = max(sims) if sims else 0.0
        
        # Spatio-temporal continuity prior & occlusion bridging
        if camera_id == self.last_camera:
            dt = frame_idx - self.last_frame
            dx = center[0] - self.last_center[0]
            dy = center[1] - self.last_center[1]
            dist = np.hypot(dx, dy)
            
            # Gated spatial continuity: only apply modest continuity bonus if appearance already matches
            if dt < 90 and raw_sim >= 0.58:
                if dist < 120:
                    return raw_sim + 0.10, raw_sim
                elif dist < 280 and abs(dy) < 100:  # Consistent walking motion across occluders
                    return raw_sim + 0.08, raw_sim
            elif dt > 180 and dist > 350:
                return raw_sim - 0.10, raw_sim
                
        return raw_sim, raw_sim

    def update(self, new_embedding: np.ndarray, camera_id: str,
               center: Tuple[int, int], frame_idx: int, timestamp: datetime,
               crop: Optional[np.ndarray] = None):
        self.total_detections += 1
        self.last_camera = camera_id
        self.last_center = center
        self.last_frame = frame_idx
        self.last_timestamp = timestamp
        
        sims = [float(np.dot(new_embedding, ex)) for ex in self.exemplars]
        raw_sim = max(sims) if sims else 0.0
        
        # Guard against gallery contamination from passing shadows / occluders
        if 0.72 <= raw_sim < 0.88 and len(self.exemplars) < config.MAX_EXEMPLARS_PER_PERSON:
            self.exemplars.append(new_embedding.copy())
            if crop is not None:
                self.save_crop(crop)
        elif raw_sim >= 0.72:
            best_idx = int(np.argmax(sims))
            updated = config.REID_FEATURE_ALPHA * self.exemplars[best_idx] + (1.0 - config.REID_FEATURE_ALPHA) * new_embedding
            self.exemplars[best_idx] = (updated / (np.linalg.norm(updated) + 1e-6)).astype(np.float32)

    def save_crop(self, crop: np.ndarray):
        if crop is None or crop.size == 0:
            return
        filename = f"{self.global_id}_sample_{len(self.crops)+1}.jpg"
        filepath = os.path.join(str(config.CROPS_DIR), filename)
        cv2.imwrite(filepath, crop)
        self.crops.append(filepath)
        if len(self.crops) > config.MAX_GALLERY_IMAGES_PER_PERSON:
            oldest = self.crops.pop(0)
            if os.path.exists(oldest):
                try:
                    os.remove(oldest)
                except Exception:
                    pass


class ReIDMemoryBank:
    def __init__(self, extractor: DeepReIDExtractor, similarity_threshold: float = config.REID_SIMILARITY_THRESHOLD):
        self.extractor = extractor
        self.similarity_threshold = similarity_threshold
        self.persons: Dict[str, PersonMemory] = {}
        self.next_id_number = 1
        self.local_to_global_cache: Dict[Tuple[str, int], str] = {}

    def _generate_global_id(self) -> str:
        gid = f"Person-{self.next_id_number:03d}"
        self.next_id_number += 1
        return gid

    def match_or_register(self, camera_id: str, local_track_id: Optional[int],
                          crop: np.ndarray, center: Tuple[int, int],
                          frame_idx: int, timestamp: datetime,
                          is_entrance: bool = False,
                          excluded_gids: Optional[Set[str]] = None) -> Tuple[str, float, bool]:
        if excluded_gids is None:
            excluded_gids = set()
            
        cache_key = (camera_id, local_track_id) if local_track_id is not None and local_track_id >= 0 else None
        
        # 1. Existing local-to-global association
        if cache_key and cache_key in self.local_to_global_cache:
            gid = self.local_to_global_cache[cache_key]
            if gid not in excluded_gids:
                if crop is not None and crop.shape[0] > 30 and crop.shape[1] > 15:
                    feat = self.extractor.extract_features([crop])
                    if feat.shape[0] > 0 and gid in self.persons:
                        self.persons[gid].update(feat[0], camera_id, center, frame_idx, timestamp, crop)
                return gid, 1.0, False

        # 2. Extract candidate feature
        feat = self.extractor.extract_features([crop])
        if feat.shape[0] == 0:
            new_gid = self._generate_global_id()
            if cache_key:
                self.local_to_global_cache[cache_key] = new_gid
            return new_gid, 0.0, True
            
        candidate_feat = feat[0]
        
        # 3. Spatio-Temporal ReID Matching
        best_gid = None
        best_score = -1.0
        
        for gid, mem in self.persons.items():
            if gid in excluded_gids:
                continue
                
            eff_score, _ = mem.match_score(candidate_feat, camera_id, center, frame_idx)
            if eff_score > best_score:
                best_score = eff_score
                best_gid = gid
                
        # 4. Decision
        if best_score >= self.similarity_threshold and best_gid is not None:
            self.persons[best_gid].update(candidate_feat, camera_id, center, frame_idx, timestamp, crop)
            if cache_key:
                self.local_to_global_cache[cache_key] = best_gid
            return best_gid, best_score, False
        else:
            new_gid = self._generate_global_id()
            self.persons[new_gid] = PersonMemory(new_gid, candidate_feat, crop, camera_id, center, frame_idx, timestamp)
            if cache_key:
                self.local_to_global_cache[cache_key] = new_gid
            return new_gid, max(0.0, best_score), True

    def get_person_crops(self, global_id: str) -> List[str]:
        if global_id in self.persons:
            return self.persons[global_id].crops
        return []