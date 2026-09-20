"""
Side Dashboard HUD Generator.
Renders a real-time analytics panel displaying mall visitor KPIs,
camera occupancy, active person tracking table, and uncropped live activity audit log.
Dynamically scales layout based on video feed height to prevent any text clipping.
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import config

class SideDashboardRenderer:
    """
    Renders the live stats dashboard panel beside CCTV video feeds.
    """
    def __init__(self, width: int = config.DASHBOARD_WIDTH):
        self.width = width
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, height: int, metrics: Dict[str, Any],
               current_time: datetime) -> np.ndarray:
        """
        Creates the dashboard image canvas with all metrics, tables, and full activity log.
        """
        panel = np.full((height, self.width, 3), config.DASHBOARD_BG_COLOR, dtype=np.uint8)
        is_compact = (height < 520)
        
        # -------------------------------------------------------------
        # 1. Header Banner
        # -------------------------------------------------------------
        if is_compact:
            y_cursor = 18
            cv2.putText(panel, "MARTIAN CCTV INTELLIGENCE", (16, y_cursor),
                        self.font, 0.48, config.TEXT_WHITE, 1, cv2.LINE_AA)
            y_cursor = 33
            time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(panel, time_str, (16, y_cursor),
                        self.font, 0.36, config.ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.circle(panel, (self.width - 24, y_cursor - 3), 4, config.ACCENT_GREEN, -1)
            cv2.putText(panel, "LIVE", (self.width - 56, y_cursor),
                        self.font, 0.34, config.ACCENT_GREEN, 1, cv2.LINE_AA)
            y_cursor = 40
            cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (50, 56, 70), 1)
            y_cursor = 46
        else:
            y_cursor = 24
            cv2.putText(panel, "MARTIAN CCTV INTELLIGENCE", (20, y_cursor),
                        self.font, 0.62, config.TEXT_WHITE, 2, cv2.LINE_AA)
            y_cursor += 18
            cv2.putText(panel, "Multi-Camera ReID & Journey Tracking", (20, y_cursor),
                        self.font, 0.42, config.TEXT_MUTED, 1, cv2.LINE_AA)
            y_cursor += 20
            time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(panel, time_str, (20, y_cursor),
                        self.font, 0.45, config.ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.circle(panel, (self.width - 30, y_cursor - 4), 5, config.ACCENT_GREEN, -1)
            cv2.putText(panel, "LIVE", (self.width - 70, y_cursor),
                        self.font, 0.40, config.ACCENT_GREEN, 1, cv2.LINE_AA)
            y_cursor += 14
            cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (50, 56, 70), 1)
            y_cursor += 16

        # -------------------------------------------------------------
        # 2. Key Metrics Cards (KPIs)
        # -------------------------------------------------------------
        total_entered = metrics.get("total_entered", 0)
        currently_present = metrics.get("currently_present", 0)
        total_exited = metrics.get("total_exited", 0)
        
        card_w = (self.width - 36) // 3
        card_h = 36 if is_compact else 50
        
        cards = [
            ("TOTAL IN", str(total_entered), config.ACCENT_CYAN),
            ("PRESENT", str(currently_present), config.ACCENT_GREEN),
            ("EXITED", str(total_exited), config.ACCENT_ORANGE),
        ]
        
        for idx, (label, val, color) in enumerate(cards):
            cx = 12 + idx * (card_w + 6)
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          config.DASHBOARD_CARD_BG, -1)
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          (60, 68, 85), 1)
            if is_compact:
                cv2.putText(panel, label, (cx + 6, y_cursor + 13),
                            self.font, 0.30, config.TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(panel, val, (cx + 8, y_cursor + 30),
                            self.font, 0.55, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, label, (cx + 8, y_cursor + 16),
                            self.font, 0.36, config.TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(panel, val, (cx + 10, y_cursor + 40),
                            self.font, 0.68, color, 2, cv2.LINE_AA)
                            
        y_cursor += card_h + (8 if is_compact else 16)
        cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (50, 56, 70), 1)
        y_cursor += (10 if is_compact else 16)

        # -------------------------------------------------------------
        # 3. Camera Occupancy Distribution
        # -------------------------------------------------------------
        cam_counts = metrics.get("camera_counts", {})
        if is_compact:
            # Single compact summary line
            cam_parts = []
            for cam_id in config.DEFAULT_CAMERAS.keys():
                cnt = cam_counts.get(cam_id, 0)
                cam_parts.append(f"{cam_id}: {cnt} active" if cnt > 0 else f"{cam_id}: 0")
            occ_line = " | ".join(cam_parts)
            cv2.putText(panel, occ_line, (16, y_cursor),
                        self.font, 0.33, (180, 190, 205), 1, cv2.LINE_AA)
            y_cursor += 8
            cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (50, 56, 70), 1)
            y_cursor += 12
        else:
            cv2.putText(panel, "CAMERA OCCUPANCY", (18, y_cursor),
                        self.font, 0.44, config.TEXT_WHITE, 1, cv2.LINE_AA)
            for cam_id, info in config.DEFAULT_CAMERAS.items():
                count = cam_counts.get(cam_id, 0)
                cam_name = info.get("name", cam_id)
                y_cursor += 18
                cv2.putText(panel, f"{cam_id}: {cam_name[:20]}", (20, y_cursor),
                            self.font, 0.38, config.TEXT_MUTED, 1, cv2.LINE_AA)
                badge_text = f"{count} active"
                (bw, _), _ = cv2.getTextSize(badge_text, self.font, 0.38, 1)
                bx = self.width - 20 - bw
                cv2.putText(panel, badge_text, (bx, y_cursor),
                            self.font, 0.38, config.ACCENT_CYAN if count > 0 else (120, 120, 120), 1, cv2.LINE_AA)
            y_cursor += 12
            cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (50, 56, 70), 1)
            y_cursor += 14

        # -------------------------------------------------------------
        # 4. Active Visitors Table
        # -------------------------------------------------------------
        cv2.putText(panel, "ACTIVE VISITORS IN PREMISES", (16, y_cursor),
                    self.font, 0.38 if is_compact else 0.44, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += (12 if is_compact else 16)
        
        # Table Header
        headers = [("ID", 16), ("ENTRY", 100), ("LOCATION", 190), ("DWELL", 320)]
        for h_text, hx in headers:
            cv2.putText(panel, h_text, (hx, y_cursor),
                        self.font, 0.30 if is_compact else 0.34, (140, 145, 155), 1, cv2.LINE_AA)
        y_cursor += 4
        cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (45, 50, 62), 1)
        
        active_list: List[Any] = metrics.get("active_visitors", [])
        max_active_display = 3 if is_compact else 5
        display_list = active_list[-max_active_display:]
        
        if not display_list:
            y_cursor += (16 if is_compact else 22)
            cv2.putText(panel, "No visitors currently detected", (16, y_cursor),
                        self.font, 0.34, (110, 115, 125), 1, cv2.LINE_AA)
        else:
            row_step = 16 if is_compact else 20
            for rec in reversed(display_list):
                y_cursor += row_step
                entry_str = rec.entry_time.strftime("%H:%M:%S")
                dwell_str = rec.dwell_time_str
                
                elapsed = (current_time - rec.last_seen_time).total_seconds()
                is_live = elapsed <= 2.0
                
                id_col = config.ACCENT_CYAN if is_live else (160, 165, 175)
                cv2.putText(panel, rec.global_id, (16, y_cursor),
                            self.font, 0.34 if is_compact else 0.38, id_col, 1, cv2.LINE_AA)
                cv2.putText(panel, entry_str, (100, y_cursor),
                            self.font, 0.32 if is_compact else 0.36, config.TEXT_WHITE if is_live else config.TEXT_MUTED, 1, cv2.LINE_AA)
                cam_str = rec.last_camera if is_live else f"{rec.last_camera} (Occl)"
                cam_col = config.TEXT_MUTED if is_live else (130, 140, 160)
                cv2.putText(panel, cam_str, (190, y_cursor),
                            self.font, 0.30 if is_compact else 0.34, cam_col, 1, cv2.LINE_AA)
                dwell_col = config.ACCENT_GREEN if is_live else (120, 170, 130)
                cv2.putText(panel, dwell_str, (320, y_cursor),
                            self.font, 0.32 if is_compact else 0.36, dwell_col, 1, cv2.LINE_AA)
                            
        y_cursor += 8
        cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (50, 56, 70), 1)
        y_cursor += (14 if is_compact else 18)

        # -------------------------------------------------------------
        # 5. Live Activity Event Ticker (Never Cropped)
        # -------------------------------------------------------------
        cv2.putText(panel, "ACTIVITY LOG (LIVE AUDIT)", (16, y_cursor),
                    self.font, 0.38 if is_compact else 0.44, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += 6
        
        events = metrics.get("recent_events", [])
        line_height = 16 if is_compact else 18
        
        # Display as many recent events as fit cleanly before the bottom boundary
        for ev in reversed(events):
            if y_cursor + line_height >= height - 6:
                break
                
            y_cursor += line_height
            ev_type = ev.get("type", "EVENT")
            ev_time = ev.get("time", "")
            raw_text = ev.get("text", "")
            
            # Format clean message without redundant timestamps
            if " [" in raw_text:
                clean_msg = raw_text.split(" [")[0]
            else:
                clean_msg = raw_text
                
            line_text = f"[{ev_time}] {clean_msg}"
            if len(line_text) > 48:
                line_text = line_text[:45] + "..."
                
            if ev_type == "ENTRY":
                color = config.ACCENT_CYAN
            elif ev_type == "EXIT":
                color = config.ACCENT_ORANGE
            elif ev_type == "RE_ENTRY":
                color = config.ACCENT_GREEN
            else:
                color = (200, 180, 255)  # Purple/Pink for TRANSITIONS
                
            cv2.putText(panel, line_text, (16, y_cursor),
                        self.font, 0.32 if is_compact else 0.36, color, 1, cv2.LINE_AA)
                        
        return panel