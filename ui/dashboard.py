"""
Side Dashboard HUD Generator.
Renders a real-time analytics panel displaying mall visitor KPIs,
camera occupancy, active person tracking table, and live event ticker.
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
        Creates the dashboard image canvas with all metrics and tables.
        """
        panel = np.full((height, self.width, 3), config.DASHBOARD_BG_COLOR, dtype=np.uint8)
        
        y_cursor = 24
        
        # 1. Header Banner
        cv2.putText(panel, "MARTIAN CCTV INTELLIGENCE", (20, y_cursor),
                    self.font, 0.62, config.TEXT_WHITE, 2, cv2.LINE_AA)
        y_cursor += 18
        cv2.putText(panel, "Multi-Camera ReID & Journey Tracking", (20, y_cursor),
                    self.font, 0.42, config.TEXT_MUTED, 1, cv2.LINE_AA)
        y_cursor += 22
        
        # Live Time & System Pulse
        time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(panel, time_str, (20, y_cursor),
                    self.font, 0.45, config.ACCENT_CYAN, 1, cv2.LINE_AA)
        
        # Green pulse status
        cv2.circle(panel, (self.width - 35, y_cursor - 4), 5, config.ACCENT_GREEN, -1)
        cv2.putText(panel, "LIVE", (self.width - 75, y_cursor),
                    self.font, 0.40, config.ACCENT_GREEN, 1, cv2.LINE_AA)
                    
        y_cursor += 16
        cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (50, 56, 70), 1)
        y_cursor += 20
        
        # 2. Key Metrics Cards (KPIs)
        total_entered = metrics.get("total_entered", 0)
        currently_present = metrics.get("currently_present", 0)
        total_exited = metrics.get("total_exited", 0)
        
        card_w = (self.width - 48) // 3
        card_h = 56
        
        cards = [
            ("TOTAL IN", str(total_entered), config.ACCENT_CYAN),
            ("PRESENT", str(currently_present), config.ACCENT_GREEN),
            ("EXITED", str(total_exited), config.ACCENT_ORANGE),
        ]
        
        for idx, (label, val, color) in enumerate(cards):
            cx = 16 + idx * (card_w + 8)
            # Card background
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          config.DASHBOARD_CARD_BG, -1)
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          (60, 68, 85), 1)
            # Card label
            cv2.putText(panel, label, (cx + 8, y_cursor + 18),
                        self.font, 0.38, config.TEXT_MUTED, 1, cv2.LINE_AA)
            # Card value
            cv2.putText(panel, val, (cx + 10, y_cursor + 44),
                        self.font, 0.72, color, 2, cv2.LINE_AA)
                        
        y_cursor += card_h + 20
        
        # 3. Camera Occupancy Distribution
        cv2.putText(panel, "CAMERA OCCUPANCY", (20, y_cursor),
                    self.font, 0.48, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += 12
        
        cam_counts = metrics.get("camera_counts", {})
        for cam_id, info in config.DEFAULT_CAMERAS.items():
            count = cam_counts.get(cam_id, 0)
            cam_name = info.get("name", cam_id)
            
            # Row container
            y_cursor += 18
            cv2.putText(panel, f"{cam_id}: {cam_name[:22]}", (24, y_cursor),
                        self.font, 0.40, config.TEXT_MUTED, 1, cv2.LINE_AA)
                        
            # Badge count
            badge_text = f"{count} active"
            (bw, _), _ = cv2.getTextSize(badge_text, self.font, 0.38, 1)
            bx = self.width - 24 - bw
            cv2.putText(panel, badge_text, (bx, y_cursor),
                        self.font, 0.40, config.ACCENT_CYAN if count > 0 else (120, 120, 120), 1, cv2.LINE_AA)
                        
        y_cursor += 20
        cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (50, 56, 70), 1)
        y_cursor += 20
        
        # 4. Active Person Tracking Table (Current Time)
        cv2.putText(panel, "ACTIVE VISITORS IN PREMISES", (20, y_cursor),
                    self.font, 0.48, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += 16
        
        # Table Header
        headers = [("ID", 20), ("ENTRY", 125), ("CAMERA", 220), ("DWELL", 325)]
        for h_text, hx in headers:
            cv2.putText(panel, h_text, (hx, y_cursor),
                        self.font, 0.36, (140, 145, 155), 1, cv2.LINE_AA)
        y_cursor += 6
        cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (45, 50, 62), 1)
        
        active_list: List[Any] = metrics.get("active_visitors", [])
        # Show top 5 active persons
        display_list = active_list[-6:]
        if not display_list:
            y_cursor += 24
            cv2.putText(panel, "No visitors currently detected", (20, y_cursor),
                        self.font, 0.40, (110, 115, 125), 1, cv2.LINE_AA)
        else:
            for rec in reversed(display_list):
                y_cursor += 22
                entry_str = rec.entry_time.strftime("%H:%M:%S")
                dwell_str = rec.dwell_time_str
                
                # Global ID
                cv2.putText(panel, rec.global_id, (20, y_cursor),
                            self.font, 0.40, config.ACCENT_CYAN, 1, cv2.LINE_AA)
                # Entry Time
                cv2.putText(panel, entry_str, (125, y_cursor),
                            self.font, 0.38, config.TEXT_WHITE, 1, cv2.LINE_AA)
                # Last Camera
                cv2.putText(panel, rec.last_camera, (220, y_cursor),
                            self.font, 0.38, config.TEXT_MUTED, 1, cv2.LINE_AA)
                # Dwell
                cv2.putText(panel, dwell_str, (325, y_cursor),
                            self.font, 0.38, config.ACCENT_GREEN, 1, cv2.LINE_AA)
                            
        # 5. Live Activity Event Ticker (at the bottom)
        ticker_y = max(y_cursor + 28, height - 120)
        cv2.line(panel, (16, ticker_y), (self.width - 16, ticker_y), (50, 56, 70), 1)
        ticker_y += 18
        
        cv2.putText(panel, "ACTIVITY LOG", (20, ticker_y),
                    self.font, 0.45, config.TEXT_WHITE, 1, cv2.LINE_AA)
                    
        events = metrics.get("recent_events", [])
        for ev in reversed(events[-4:]):
            ticker_y += 18
            ev_type = ev.get("type", "EVENT")
            color = config.ACCENT_CYAN if ev_type == "ENTRY" else (config.ACCENT_ORANGE if ev_type == "EXIT" else config.TEXT_MUTED)
            text = f"[{ev.get('time', '')}] {ev.get('text', '')}"
            if len(text) > 42:
                text = text[:39] + "..."
            cv2.putText(panel, text, (20, ticker_y),
                        self.font, 0.36, color, 1, cv2.LINE_AA)
                        
        return panel