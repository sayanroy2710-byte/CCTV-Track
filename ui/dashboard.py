"""
Side Dashboard HUD Generator with High-Clarity TrueType Vector Rendering.
Renders real-time visitor KPIs, camera occupancy, active person tracking table,
and crystal-clear activity log cards using anti-aliased TrueType fonts (Segoe UI Bold).
Eliminates text haziness, pixelation, and small blurry fonts across all screen sizes.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import config

class SideDashboardRenderer:
    """
    Renders the live stats dashboard panel beside CCTV video feeds with razor-sharp readability.
    """
    def __init__(self, width: int = config.DASHBOARD_WIDTH):
        self.width = width
        self._init_fonts()

    def _init_fonts(self):
        # Locate system TrueType font for crystal-clear vector glyphs
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",  # Windows Segoe UI Bold
            "C:/Windows/Fonts/arialbd.ttf",   # Windows Arial Bold
            "arial.ttf"
        ]
        chosen_font = None
        for p in font_candidates:
            if os.path.exists(p):
                chosen_font = p
                break
                
        try:
            if chosen_font:
                self.f_title = ImageFont.truetype(chosen_font, 22)
                self.f_sub = ImageFont.truetype(chosen_font, 14)
                self.f_time = ImageFont.truetype(chosen_font, 17)
                self.f_card_lbl = ImageFont.truetype(chosen_font, 13)
                self.f_card_val = ImageFont.truetype(chosen_font, 30)
                self.f_sec_hdr = ImageFont.truetype(chosen_font, 15)
                self.f_tbl_hdr = ImageFont.truetype(chosen_font, 13)
                self.f_tbl_row = ImageFont.truetype(chosen_font, 14)
                self.f_badge = ImageFont.truetype(chosen_font, 12)
                self.f_log = ImageFont.truetype(chosen_font, 14)
            else:
                raise ValueError("No TrueType font file found")
        except Exception:
            # Fallback to default
            self.f_title = ImageFont.load_default()
            self.f_sub = ImageFont.load_default()
            self.f_time = ImageFont.load_default()
            self.f_card_lbl = ImageFont.load_default()
            self.f_card_val = ImageFont.load_default()
            self.f_sec_hdr = ImageFont.load_default()
            self.f_tbl_hdr = ImageFont.load_default()
            self.f_tbl_row = ImageFont.load_default()
            self.f_badge = ImageFont.load_default()
            self.f_log = ImageFont.load_default()

    def render(self, height: int, metrics: Dict[str, Any],
               current_time: datetime) -> np.ndarray:
        """
        Creates the dashboard image canvas with all metrics, tables, and high-contrast event cards.
        """
        w = self.width
        h = height
        img = Image.new("RGB", (w, h), (24, 28, 36))
        draw = ImageDraw.Draw(img)
        
        # 1. Header Banner
        draw.text((20, 16), "MARTIAN CCTV INTELLIGENCE", font=self.f_title, fill=(255, 255, 255))
        draw.text((20, 44), "Multi-Camera ReID & Footfall Analytics", font=self.f_sub, fill=(160, 165, 175))
        
        time_str = current_time.strftime("%Y-%m-%d  %H:%M:%S")
        draw.text((20, 68), time_str, font=self.f_time, fill=(0, 210, 255))
        
        # LIVE indicator pill
        draw.rounded_rectangle((w - 78, 66, w - 18, 90), radius=5, fill=(20, 70, 35), outline=(50, 220, 80), width=1)
        draw.text((w - 65, 70), "LIVE", font=self.f_badge, fill=(80, 255, 120))
        draw.line((16, 98, w - 16, 98), fill=(50, 56, 70), width=1)
        
        # 2. KPI Cards
        total_entered = metrics.get("total_entered", 0)
        currently_present = metrics.get("currently_present", 0)
        total_exited = metrics.get("total_exited", 0)
        
        cw = (w - 48) // 3
        cards = [
            ("TOTAL IN", str(total_entered), (0, 210, 255)),
            ("PRESENT", str(currently_present), (80, 255, 120)),
            ("EXITED", str(total_exited), (255, 145, 0)),
        ]
        for idx, (lbl, val, col) in enumerate(cards):
            cx = 16 + idx * (cw + 8)
            draw.rounded_rectangle((cx, 110, cx + cw, 170), radius=6, fill=(36, 42, 54), outline=(60, 68, 85), width=1)
            draw.text((cx + 10, 116), lbl, font=self.f_card_lbl, fill=(160, 165, 175))
            draw.text((cx + 12, 134), val, font=self.f_card_val, fill=col)
        draw.line((16, 182, w - 16, 182), fill=(50, 56, 70), width=1)
        
        # 3. Active Visitors Table
        draw.text((20, 194), "ACTIVE VISITORS IN PREMISES", font=self.f_sec_hdr, fill=(255, 255, 255))
        draw.text((20, 218), "ID", font=self.f_tbl_hdr, fill=(140, 145, 155))
        draw.text((120, 218), "ENTRY", font=self.f_tbl_hdr, fill=(140, 145, 155))
        draw.text((215, 218), "LOCATION", font=self.f_tbl_hdr, fill=(140, 145, 155))
        draw.text((360, 218), "DWELL", font=self.f_tbl_hdr, fill=(140, 145, 155))
        draw.line((16, 238, w - 16, 238), fill=(45, 50, 62), width=1)
        
        active_list: List[Any] = metrics.get("active_visitors", [])
        display_list = active_list[-4:]
        vy = 244
        if not display_list:
            draw.text((20, vy), "No visitors currently detected", font=self.f_tbl_row, fill=(120, 125, 135))
            vy += 22
        else:
            for rec in reversed(display_list):
                elapsed = (current_time - rec.last_seen_time).total_seconds()
                is_live = elapsed <= 2.0
                id_col = (0, 210, 255) if is_live else (160, 165, 175)
                time_col = (240, 240, 240) if is_live else (150, 155, 165)
                cam_col = (180, 190, 205) if is_live else (130, 135, 145)
                dwell_col = (80, 255, 120) if is_live else (120, 170, 130)
                
                cam_str = rec.last_camera if is_live else f"{rec.last_camera} (Occl)"
                
                draw.text((20, vy), rec.global_id, font=self.f_tbl_row, fill=id_col)
                draw.text((120, vy), rec.entry_time.strftime("%H:%M:%S"), font=self.f_tbl_row, fill=time_col)
                draw.text((215, vy), cam_str, font=self.f_tbl_row, fill=cam_col)
                draw.text((360, vy), rec.dwell_time_str, font=self.f_tbl_row, fill=dwell_col)
                vy += 22
        draw.line((16, vy + 6, w - 16, vy + 6), fill=(50, 56, 70), width=1)
        
        # 4. Activity Log Cards (Bold, Large, Crystal-Clear)
        ly = vy + 18
        draw.text((20, ly), "ACTIVITY LOG (REAL-TIME AUDIT)", font=self.f_sec_hdr, fill=(255, 255, 255))
        ly += 24
        
        events = metrics.get("recent_events", [])
        for ev in reversed(events):
            if ly + 36 >= h - 8:
                break
                
            ev_type = ev.get("type", "EVENT")
            ev_time = ev.get("time", "")
            raw_text = ev.get("text", "")
            if " [" in raw_text:
                clean_msg = raw_text.split(" [")[0]
            else:
                clean_msg = raw_text
                
            parts = clean_msg.split(" ", 1)
            pid = parts[0] if len(parts) > 0 else ""
            desc = parts[1] if len(parts) > 1 else ""
            
            if ev_type == "ENTRY":
                badge_bg = (0, 210, 255)
                badge_fg = (0, 0, 0)
                badge_tag = "ENTRY"
            elif ev_type == "EXIT":
                badge_bg = (255, 145, 0)
                badge_fg = (0, 0, 0)
                badge_tag = "EXIT"
            elif ev_type == "RE_ENTRY":
                badge_bg = (80, 255, 120)
                badge_fg = (0, 0, 0)
                badge_tag = "RE-ID"
            else:
                badge_bg = (255, 128, 171)
                badge_fg = (0, 0, 0)
                badge_tag = "MOVE"
                
            # Card container
            draw.rounded_rectangle((16, ly, w - 16, ly + 32), radius=5, fill=(32, 38, 50), outline=(55, 64, 82), width=1)
            # Left accent stripe
            draw.rectangle((16, ly, 21, ly + 32), fill=badge_bg)
            # Pill badge
            draw.rounded_rectangle((28, ly + 5, 82, ly + 27), radius=4, fill=badge_bg)
            draw.text((36, ly + 8), badge_tag, font=self.f_badge, fill=badge_fg)
            # Timestamp
            draw.text((90, ly + 7), ev_time, font=self.f_log, fill=(160, 205, 255))
            # Person ID
            draw.text((165, ly + 7), f"{pid}:", font=self.f_log, fill=(255, 255, 255))
            # Description
            draw.text((260, ly + 7), desc, font=self.f_log, fill=(230, 235, 245))
            ly += 36

        # Convert back to BGR numpy array for OpenCV
        panel_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return panel_bgr