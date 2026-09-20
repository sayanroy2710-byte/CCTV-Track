"""
Side Dashboard HUD Generator.
Renders a high-definition real-time analytics panel displaying mall visitor KPIs,
camera occupancy, active person tracking table, and crystal-clear activity log cards.
Designed with enterprise-grade typography, distinct colored pill badges, and high-contrast
card containers for effortless readability in exported videos and live display windows.
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import config

class SideDashboardRenderer:
    """
    Renders the live stats dashboard panel beside CCTV video feeds with razor-sharp readability.
    """
    def __init__(self, width: int = config.DASHBOARD_WIDTH):
        self.width = width
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, height: int, metrics: Dict[str, Any],
               current_time: datetime) -> np.ndarray:
        """
        Creates the dashboard image canvas with all metrics, tables, and high-contrast event cards.
        """
        panel = np.full((height, self.width, 3), config.DASHBOARD_BG_COLOR, dtype=np.uint8)
        is_compact = (height < 600)
        
        # -------------------------------------------------------------
        # 1. Header Banner
        # -------------------------------------------------------------
        if is_compact:
            y_cursor = 22
            cv2.putText(panel, "MARTIAN CCTV INTELLIGENCE", (16, y_cursor),
                        self.font, 0.52, config.TEXT_WHITE, 2, cv2.LINE_AA)
            y_cursor = 40
            time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(panel, time_str, (16, y_cursor),
                        self.font, 0.40, config.ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.circle(panel, (self.width - 24, y_cursor - 4), 5, config.ACCENT_GREEN, -1)
            cv2.putText(panel, "LIVE", (self.width - 62, y_cursor),
                        self.font, 0.38, config.ACCENT_GREEN, 1, cv2.LINE_AA)
            y_cursor = 48
            cv2.line(panel, (12, y_cursor), (self.width - 12, y_cursor), (50, 56, 70), 1)
            y_cursor = 56
        else:
            y_cursor = 32
            cv2.putText(panel, "MARTIAN CCTV INTELLIGENCE", (20, y_cursor),
                        self.font, 0.65, config.TEXT_WHITE, 2, cv2.LINE_AA)
            y_cursor = 54
            cv2.putText(panel, "Multi-Camera ReID & Footfall Analytics", (20, y_cursor),
                        self.font, 0.42, config.TEXT_MUTED, 1, cv2.LINE_AA)
            y_cursor = 78
            time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(panel, time_str, (20, y_cursor),
                        self.font, 0.46, config.ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.circle(panel, (self.width - 28, y_cursor - 5), 6, config.ACCENT_GREEN, -1)
            cv2.putText(panel, "LIVE", (self.width - 72, y_cursor),
                        self.font, 0.42, config.ACCENT_GREEN, 1, cv2.LINE_AA)
            y_cursor = 88
            cv2.line(panel, (16, y_cursor), (self.width - 16, y_cursor), (50, 56, 70), 1)
            y_cursor = 100

        # -------------------------------------------------------------
        # 2. Key Metrics Cards (KPIs)
        # -------------------------------------------------------------
        total_entered = metrics.get("total_entered", 0)
        currently_present = metrics.get("currently_present", 0)
        total_exited = metrics.get("total_exited", 0)
        
        card_w = (self.width - (32 if is_compact else 48)) // 3
        card_h = 42 if is_compact else 58
        
        cards = [
            ("TOTAL IN", str(total_entered), config.ACCENT_CYAN),
            ("PRESENT", str(currently_present), config.ACCENT_GREEN),
            ("EXITED", str(total_exited), config.ACCENT_ORANGE),
        ]
        
        for idx, (label, val, color) in enumerate(cards):
            cx = (12 if is_compact else 16) + idx * (card_w + (6 if is_compact else 8))
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          config.DASHBOARD_CARD_BG, -1)
            cv2.rectangle(panel, (cx, y_cursor), (cx + card_w, y_cursor + card_h),
                          (60, 68, 85), 1)
            if is_compact:
                cv2.putText(panel, label, (cx + 8, y_cursor + 14),
                            self.font, 0.32, config.TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(panel, val, (cx + 10, y_cursor + 34),
                            self.font, 0.62, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(panel, label, (cx + 10, y_cursor + 18),
                            self.font, 0.36, config.TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(panel, val, (cx + 12, y_cursor + 46),
                            self.font, 0.78, color, 2, cv2.LINE_AA)
                            
        y_cursor += card_h + (10 if is_compact else 14)
        cv2.line(panel, (12 if is_compact else 16, y_cursor),
                 (self.width - (12 if is_compact else 16), y_cursor), (50, 56, 70), 1)
        y_cursor += (12 if is_compact else 18)

        # -------------------------------------------------------------
        # 3. Active Visitors Table
        # -------------------------------------------------------------
        title_font = 0.42 if is_compact else 0.46
        cv2.putText(panel, "ACTIVE VISITORS IN PREMISES", (16 if is_compact else 20, y_cursor),
                    self.font, title_font, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += (16 if is_compact else 22)
        
        # Table Header
        h_font = 0.34 if is_compact else 0.38
        headers = [("ID", 16 if is_compact else 20),
                   ("ENTRY", 110 if is_compact else 125),
                   ("LOCATION", 205 if is_compact else 225),
                   ("DWELL", 345 if is_compact else 370)]
        for h_text, hx in headers:
            cv2.putText(panel, h_text, (hx, y_cursor),
                        self.font, h_font, (140, 145, 155), 1, cv2.LINE_AA)
        y_cursor += (5 if is_compact else 6)
        cv2.line(panel, (12 if is_compact else 16, y_cursor),
                 (self.width - (12 if is_compact else 16), y_cursor), (45, 50, 62), 1)
        
        active_list: List[Any] = metrics.get("active_visitors", [])
        max_active_display = 3 if is_compact else 4
        display_list = active_list[-max_active_display:]
        
        if not display_list:
            y_cursor += (18 if is_compact else 24)
            cv2.putText(panel, "No visitors currently in camera view", (16 if is_compact else 20, y_cursor),
                        self.font, 0.36 if is_compact else 0.40, (110, 115, 125), 1, cv2.LINE_AA)
        else:
            row_step = 18 if is_compact else 24
            for rec in reversed(display_list):
                y_cursor += row_step
                entry_str = rec.entry_time.strftime("%H:%M:%S")
                dwell_str = rec.dwell_time_str
                
                elapsed = (current_time - rec.last_seen_time).total_seconds()
                is_live = elapsed <= 2.0
                
                id_col = config.ACCENT_CYAN if is_live else (160, 165, 175)
                row_f = 0.36 if is_compact else 0.42
                cv2.putText(panel, rec.global_id, (16 if is_compact else 20, y_cursor),
                            self.font, row_f, id_col, 1, cv2.LINE_AA)
                cv2.putText(panel, entry_str, (110 if is_compact else 125, y_cursor),
                            self.font, row_f, config.TEXT_WHITE if is_live else config.TEXT_MUTED, 1, cv2.LINE_AA)
                cam_str = rec.last_camera if is_live else f"{rec.last_camera} (Occl)"
                cam_col = (180, 190, 205) if is_live else (130, 140, 160)
                cv2.putText(panel, cam_str, (205 if is_compact else 225, y_cursor),
                            self.font, row_f, cam_col, 1, cv2.LINE_AA)
                dwell_col = config.ACCENT_GREEN if is_live else (120, 170, 130)
                cv2.putText(panel, dwell_str, (345 if is_compact else 370, y_cursor),
                            self.font, row_f, dwell_col, 1, cv2.LINE_AA)
                            
        y_cursor += (10 if is_compact else 14)
        cv2.line(panel, (12 if is_compact else 16, y_cursor),
                 (self.width - (12 if is_compact else 16), y_cursor), (50, 56, 70), 1)
        y_cursor += (16 if is_compact else 22)

        # -------------------------------------------------------------
        # 4. Activity Log Cards (Razor-Sharp, Uncropped, High Contrast)
        # -------------------------------------------------------------
        cv2.putText(panel, "ACTIVITY LOG (REAL-TIME AUDIT)", (16 if is_compact else 20, y_cursor),
                    self.font, 0.44 if is_compact else 0.48, config.TEXT_WHITE, 1, cv2.LINE_AA)
        y_cursor += (12 if is_compact else 16)
        
        events = metrics.get("recent_events", [])
        card_step = 28 if is_compact else 34
        card_margin = 12 if is_compact else 16
        
        for ev in reversed(events):
            if y_cursor + card_step >= height - 8:
                break
                
            y_cursor += card_step
            ev_type = ev.get("type", "EVENT")
            ev_time = ev.get("time", "")
            raw_text = ev.get("text", "")
            
            # Format clean message without redundant timestamps
            if " [" in raw_text:
                clean_msg = raw_text.split(" [")[0]
            else:
                clean_msg = raw_text
                
            # Parse person and action
            parts = clean_msg.split(" ", 1)
            pid = parts[0] if len(parts) > 0 else ""
            desc = parts[1] if len(parts) > 1 else ""
            
            # Badge & accent color
            if ev_type == "ENTRY":
                accent_col = config.ACCENT_CYAN
                badge_tag = "ENTRY"
            elif ev_type == "EXIT":
                accent_col = config.ACCENT_ORANGE
                badge_tag = "EXIT"
            elif ev_type == "RE_ENTRY":
                accent_col = config.ACCENT_GREEN
                badge_tag = "RE-ID"
            else:
                accent_col = (255, 128, 171)
                badge_tag = "MOVE"
                
            card_top = y_cursor - (20 if is_compact else 24)
            card_bot = y_cursor + (6 if is_compact else 6)
            
            # 1. Dark Container Card Background
            cv2.rectangle(panel, (card_margin, card_top),
                          (self.width - card_margin, card_bot), (32, 38, 50), -1)
            cv2.rectangle(panel, (card_margin, card_top),
                          (self.width - card_margin, card_bot), (50, 58, 76), 1)
                          
            # 2. Left Colored Accent Stripe
            cv2.rectangle(panel, (card_margin, card_top),
                          (card_margin + 4, card_bot), accent_col, -1)
                          
            # 3. Pill Badge
            bw = 48 if is_compact else 54
            bx = card_margin + 8
            cv2.rectangle(panel, (bx, card_top + 3), (bx + bw, card_bot - 3), accent_col, -1)
            cv2.putText(panel, badge_tag, (bx + (4 if is_compact else 6), y_cursor - (3 if is_compact else 4)),
                        self.font, 0.32 if is_compact else 0.35, (0, 0, 0), 1, cv2.LINE_AA)
                        
            # 4. Timestamp & ID
            tx = bx + bw + 8
            cv2.putText(panel, ev_time, (tx, y_cursor - (3 if is_compact else 4)),
                        self.font, 0.34 if is_compact else 0.38, (160, 205, 255), 1, cv2.LINE_AA)
                        
            px = tx + (54 if is_compact else 64)
            cv2.putText(panel, f"{pid}:", (px, y_cursor - (3 if is_compact else 4)),
                        self.font, 0.36 if is_compact else 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                        
            # 5. Event Action Description
            dx = px + (72 if is_compact else 80)
            avail_w = (self.width - card_margin) - dx
            (tw, _), _ = cv2.getTextSize(desc, self.font, 0.34 if is_compact else 0.38, 1)
            while tw > avail_w and len(desc) > 8:
                desc = desc[:-4] + "..."
                (tw, _), _ = cv2.getTextSize(desc, self.font, 0.34 if is_compact else 0.38, 1)
                
            cv2.putText(panel, desc, (dx, y_cursor - (3 if is_compact else 4)),
                        self.font, 0.34 if is_compact else 0.38, (220, 225, 235), 1, cv2.LINE_AA)
                        
        return panel