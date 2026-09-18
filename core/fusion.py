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
        
        # Expected max radius scales with elapsed frames
        expected_radius = max(30.0, 18.0 * min(dt_frames, 10))
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
        # 1. Determine dynamic weights based on context
        if dt_frames <= 5 and ambiguity_margin >= 0.08:
            # Case A: Continuous smooth tracking, appearance is unambiguous
            # Trust motion heavily for frame-to-frame smoothness
            w_mot, w_app, w_struct = 0.45, 0.40, 0.15
            strategy = "continuous_tracking"
        elif dt_frames > 5 and ambiguity_margin >= 0.08:
            # Case B: Recovery from brief occlusion / lost track
            # Motion prediction is degraded; appearance is the primary recovery anchor
            w_mot, w_app, w_struct = 0.15, 0.55, 0.30
            strategy = "occlusion_recovery"
        else:
            # Case C: Ambiguous appearance (e.g. multiple people in similar clothing)
            # Fall back on physical body proportions and structural pose geometry
            w_mot, w_app, w_struct = 0.20, 0.35, 0.45
            strategy = "ambiguity_fallback"
            
        # Normalize weights
        total_w = w_mot + w_app + w_struct
        w_mot /= total_w
        w_app /= total_w
        w_struct /= total_w
        
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
