"""
Confidence-Weighted Multi-Cue Fusion Engine.
Combines motion consistency, deep appearance embeddings, and
pose-derived structural body proportions into a unified,
explainable identity association decision.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any

class MultiCueFusionEngine:
    """
    Fuses Motion, Appearance, and Structural Pose cues with dynamic
    confidence weighting based on occlusion and ambiguity states.
    """
    def __init__(self,
                 high_thresh: float = 0.68,
                 medium_thresh: float = 0.52):
        self.high_thresh = high_thresh
        self.medium_thresh = medium_thresh

    def compute_motion_score(self, current_pos: Tuple[int, int],
                             last_pos: Tuple[int, int],
                             dt_frames: int) -> float:
        """
        Calculates motion consistency score using Gaussian spatial decay.
        dx, dy scaled by expected maximum walking speed (approx 20px/frame at 24fps).
        """
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        dist = np.hypot(dx, dy)
        
        # Expected max radius based on realistic human walking speed in CCTV (3-5 px/frame at 24fps)
        expected_radius = max(30.0, 5.0 * min(dt_frames, 30))
        motion_sim = float(np.exp(-0.5 * (dist / expected_radius) ** 2))
        return float(np.clip(motion_sim, 0.0, 1.0))

    def compute_structural_score(self, struct_feat1: Optional[np.ndarray],
                                 struct_feat2: Optional[np.ndarray]) -> float:
        """
        Calculates cosine similarity between normalized 4-D structural vectors.
        """
        if struct_feat1 is None or struct_feat2 is None:
            return 0.50  # Neutral prior if pose not available
            
        dot = float(np.dot(struct_feat1, struct_feat2))
        # Map cosine [-1, 1] to [0, 1]
        score = (dot + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def fuse_cues(self,
                  s_motion: float,
                  s_appearance: float,
                  s_structure: float,
                  dt_frames: int,
                  ambiguity_margin: float = 1.0) -> Tuple[float, str, Dict[str, Any]]:
        """
        Dynamically weights and combines scores based on tracking context.
        Returns:
            fused_score: float in [0, 1]
            tier: 'HIGH', 'MEDIUM', or 'LOW'
            explanation: detailed breakdown dictionary
        """
        # Physically impossible teleportation guard:
        # Within the same short observation window (<= 30 frames / 1.2s), a person cannot teleport across distant zones
        if dt_frames <= 30 and s_motion < 0.02:
            return 0.0, "LOW", {"strategy": "impossible_teleportation", "fused_score": 0.0}

        # 1. Determine dynamic weights based on context
        if dt_frames <= 5 and ambiguity_margin >= 0.08:
            # Case A: Continuous smooth tracking, appearance is unambiguous
            # Trust motion heavily for frame-to-frame smoothness
            w_mot, w_app, w_struct = 0.45, 0.40, 0.15
            strategy = "continuous_tracking"
        elif dt_frames > 25:
            # Case B1: Re-entry, long absence, or cross-camera transition
            # Motion has zero bearing; appearance learned from previous clicked images is decisive
            w_mot, w_app, w_struct = 0.00, 0.80, 0.20
            strategy = "visual_reidentification"
        elif dt_frames > 5 and ambiguity_margin >= 0.08:
            # Case B2: Recovery from brief occlusion (walking behind pillar/counter)
            w_mot, w_app, w_struct = 0.10, 0.65, 0.25
            strategy = "occlusion_recovery"
        else:
            # Case C: Ambiguous appearance (multiple people in similar clothing)
            # Fall back on physical body proportions and structural pose geometry
            w_mot, w_app, w_struct = 0.10, 0.45, 0.45
            strategy = "ambiguity_fallback"
            
        # Dynamically normalize over only the observable cues
        has_struct = (s_structure > 0.0)
        has_mot = (s_motion > 0.0 and dt_frames <= 15)
        
        if not has_struct:
            w_struct = 0.0
        if not has_mot:
            w_mot = 0.0
            
        total_w = w_mot + w_app + w_struct
        if total_w > 0:
            w_mot /= total_w
            w_app /= total_w
            w_struct /= total_w
        else:
            w_app = 1.0
        
        # 2. Compute fused confidence
        fused_score = (w_mot * s_motion) + (w_app * s_appearance) + (w_struct * s_structure)
        fused_score = float(np.clip(fused_score, 0.0, 1.0))
        
        # 3. Categorize into confidence tiers
        if fused_score >= self.high_thresh:
            tier = "HIGH"
        elif fused_score >= self.medium_thresh:
            tier = "MEDIUM"
        else:
            tier = "LOW"
            
        explanation = {
            "fused_score": fused_score,
            "tier": tier,
            "strategy": strategy,
            "weights": {"motion": w_mot, "appearance": w_app, "structure": w_struct},
            "scores": {"motion": s_motion, "appearance": s_appearance, "structure": s_structure},
            "dt_frames": dt_frames,
            "ambiguity_margin": ambiguity_margin
        }
        
        return fused_score, tier, explanation
