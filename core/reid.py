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
from typing import Dict, List, Tuple, Optional, Set, Any
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


from core.fusion import MultiCueFusionEngine
from core.crop_quality import CropQualityGate

class PersonMemory:
    def __init__(self, global_id: str, initial_embedding: np.ndarray, initial_crop: np.ndarray,
                 camera_id: str, center: Tuple[int, int], frame_idx: int, timestamp: datetime,
                 initial_struct: Optional[np.ndarray] = None):
        self.global_id = global_id
        self.exemplars: List[np.ndarray] = [initial_embedding.copy()]
        self.structural_exemplars: List[np.ndarray] = [initial_struct.copy()] if initial_struct is not None else []
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
        if not sims:
            return 0.0, 0.0
            
        raw_sim = max(sims)
        visual_sim = raw_sim
        
        # Local spatio-temporal continuity boost for immediate adjacent frames
        if camera_id == self.last_camera:
            dt = frame_idx - self.last_frame
            dx = center[0] - self.last_center[0]
            dy = center[1] - self.last_center[1]
            dist = np.hypot(dx, dy)
            
            # Trajectory continuity across brief occlusions (e.g. walking behind counters/pillars)
            if dt < 45 and raw_sim >= 0.62:
                if dist < 65:
                    return visual_sim + 0.20, raw_sim
                elif dist < (20 + dt * 2.0) and abs(dy) < 40:  # Physically realistic walking speed
                    return visual_sim + 0.15, raw_sim
            # Note: No time-decay or waiting-timer penalty is applied for dt > 120.
            # Long-term re-identification is driven by pure visual match from previous clicked images.
                
        return visual_sim, raw_sim

    def get_best_structural(self) -> Optional[np.ndarray]:
        if self.structural_exemplars:
            return self.structural_exemplars[0]
        return None

    def update(self, new_embedding: np.ndarray, camera_id: str,
               center: Tuple[int, int], frame_idx: int, timestamp: datetime,
               crop: Optional[np.ndarray] = None,
               structural_feat: Optional[np.ndarray] = None):
        self.total_detections += 1
        self.last_camera = camera_id
        self.last_center = center
        self.last_frame = frame_idx
        self.last_timestamp = timestamp
        
        sims = [float(np.dot(new_embedding, ex)) for ex in self.exemplars]
        raw_sim = max(sims) if sims else 0.0
        
        # Guard against gallery contamination: only add/update when sample is confident
        if 0.72 <= raw_sim < 0.88 and len(self.exemplars) < config.MAX_EXEMPLARS_PER_PERSON:
            if crop is not None:
                self.exemplars.append(new_embedding.copy())
                self.save_crop(crop)
        elif raw_sim >= 0.72 and crop is not None:
            best_idx = int(np.argmax(sims))
            updated = config.REID_FEATURE_ALPHA * self.exemplars[best_idx] + (1.0 - config.REID_FEATURE_ALPHA) * new_embedding
            self.exemplars[best_idx] = (updated / (np.linalg.norm(updated) + 1e-6)).astype(np.float32)

        # Update structural proportion memory with exponential moving average
        if structural_feat is not None:
            if not self.structural_exemplars:
                self.structural_exemplars.append(structural_feat.copy())
            elif raw_sim >= 0.70:
                cur_s = self.structural_exemplars[0]
                up_s = 0.85 * cur_s + 0.15 * structural_feat
                self.structural_exemplars[0] = (up_s / (np.linalg.norm(up_s) + 1e-6)).astype(np.float32)

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
        self.quality_gate = CropQualityGate()
        self.fusion_engine = MultiCueFusionEngine(
            high_thresh=getattr(config, "FUSION_HIGH_CONF_THRESHOLD", 0.68),
            medium_thresh=getattr(config, "FUSION_MEDIUM_CONF_THRESHOLD", 0.52)
        )

    def _generate_global_id(self) -> str:
        gid = f"Person-{self.next_id_number:03d}"
        self.next_id_number += 1
        return gid

    def match_or_register(self, camera_id: str, local_track_id: Optional[int],
                          crop: np.ndarray, center: Tuple[int, int],
                          frame_idx: int, timestamp: datetime,
                          is_entrance: bool = False,
                          excluded_gids: Optional[Set[str]] = None,
                          structural_feat: Optional[np.ndarray] = None,
                          bbox: Optional[Tuple[int, int, int, int]] = None,
                          frame_shape: Optional[Tuple[int, ...]] = None) -> Tuple[str, float, bool]:
        if excluded_gids is None:
            excluded_gids = set()
            
        cache_key = (camera_id, local_track_id) if local_track_id is not None and local_track_id >= 0 else None
        
        # Check crop quality
        crop_usable = True
        if bbox is not None and frame_shape is not None:
            crop_usable, _, _ = self.quality_gate.assess_crop(crop, bbox, frame_shape)
        crop_for_gallery = crop if crop_usable else None

        # 1. Existing local-to-global association
        if cache_key and cache_key in self.local_to_global_cache:
            gid = self.local_to_global_cache[cache_key]
            if gid not in excluded_gids:
                if crop is not None and crop.shape[0] > 30 and crop.shape[1] > 15:
                    feat = self.extractor.extract_features([crop])
                    if feat.shape[0] > 0 and gid in self.persons:
                        cached_sims = [float(np.dot(feat[0], ex)) for ex in self.persons[gid].exemplars]
                        # Cache sanity check: if visual similarity completely collapsed (< 0.45),
                        # BoT-SORT switched identity to a different individual. Invalidate cache!
                        if cached_sims and max(cached_sims) < 0.45:
                            del self.local_to_global_cache[cache_key]
                        else:
                            self.persons[gid].update(feat[0], camera_id, center, frame_idx, timestamp, crop_for_gallery, structural_feat)
                            return gid, 1.0, False

        # 2. Extract candidate appearance feature
        feat = self.extractor.extract_features([crop])
        if feat.shape[0] == 0:
            new_gid = self._generate_global_id()
            if cache_key:
                self.local_to_global_cache[cache_key] = new_gid
            return new_gid, 0.0, True
            
        candidate_feat = feat[0]
        
        # 3. Multi-Cue Matching across active identities
        candidate_matches = []
        for gid, mem in self.persons.items():
            if gid in excluded_gids:
                continue
                
            eff_app_score, raw_app_score = mem.match_score(candidate_feat, camera_id, center, frame_idx)
            dt = max(1, frame_idx - mem.last_frame)
            s_motion = self.fusion_engine.compute_motion_score(center, mem.last_center, dt)
            s_struct = self.fusion_engine.compute_structural_score(structural_feat, mem.get_best_structural())
            
            candidate_matches.append({
                "gid": gid,
                "mem": mem,
                "eff_app": eff_app_score,
                "raw_app": raw_app_score,
                "s_motion": s_motion,
                "s_struct": s_struct,
                "dt": dt
            })
            
        if not candidate_matches:
            new_gid = self._generate_global_id()
            self.persons[new_gid] = PersonMemory(new_gid, candidate_feat, crop, camera_id, center, frame_idx, timestamp, structural_feat)
            if cache_key:
                self.local_to_global_cache[cache_key] = new_gid
            return new_gid, 0.0, True

        # Sort by appearance score to measure ambiguity margin
        candidate_matches.sort(key=lambda x: x["eff_app"], reverse=True)
        top1 = candidate_matches[0]
        ambiguity_margin = (top1["eff_app"] - candidate_matches[1]["eff_app"]) if len(candidate_matches) > 1 else 1.0

        # Score and fuse all candidates
        best_gid = None
        best_fused_score = -1.0
        best_tier = "LOW"
        
        for cand in candidate_matches:
            fused_score, tier, _ = self.fusion_engine.fuse_cues(
                s_motion=cand["s_motion"],
                s_appearance=cand["eff_app"],
                s_structure=cand["s_struct"],
                dt_frames=cand["dt"],
                ambiguity_margin=ambiguity_margin
            )
            if fused_score > best_fused_score:
                best_fused_score = fused_score
                best_gid = cand["gid"]
                best_tier = tier

        # 4. Confidence-Tiered Decision
        if best_fused_score >= self.similarity_threshold and best_gid is not None and best_tier in ("HIGH", "MEDIUM"):
            self.persons[best_gid].update(candidate_feat, camera_id, center, frame_idx, timestamp, crop_for_gallery, structural_feat)
            if cache_key:
                self.local_to_global_cache[cache_key] = best_gid
            return best_gid, best_fused_score, False
        else:
            new_gid = self._generate_global_id()
            self.persons[new_gid] = PersonMemory(new_gid, candidate_feat, crop, camera_id, center, frame_idx, timestamp, structural_feat)
            if cache_key:
                self.local_to_global_cache[cache_key] = new_gid
            return new_gid, max(0.0, best_fused_score), True

    def get_person_crops(self, global_id: str) -> List[str]:
        if global_id in self.persons:
            return self.persons[global_id].crops
        return []

    def reidentify_from_image(self, crop: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Takes an arbitrary clicked or uploaded image crop, extracts its visual embedding,
        and matches against all learned identities in the memory gallery.
        Returns top-k matching persons sorted by visual similarity score.
        """
        if crop is None or crop.size == 0:
            return []
            
        feat = self.extractor.extract_features([crop])
        if feat.shape[0] == 0:
            return []
            
        candidate_feat = feat[0]
        results = []
        for gid, mem in self.persons.items():
            sims = [float(np.dot(candidate_feat, ex)) for ex in mem.exemplars]
            if not sims:
                continue
            max_sim = max(sims)
            avg_top2 = float(np.mean(sorted(sims, reverse=True)[:2])) if len(sims) >= 2 else max_sim
            score = 0.75 * max_sim + 0.25 * avg_top2
            results.append({
                "global_id": gid,
                "similarity_score": round(score, 4),
                "confidence_pct": round(score * 100, 1),
                "last_camera": mem.last_camera,
                "total_detections": mem.total_detections,
                "thumbnails": mem.crops
            })
            
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]