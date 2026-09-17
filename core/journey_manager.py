"""
Journey Manager and Dwell-Time Tracking Engine.
Maintains state of all visitors across cameras, detects entrance,
inter-camera transitions, exit time, and computes total dwell time.
Persists all records to SQLite database (cctv_mall_tracking.db).
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import config

class VisitorRecord:
    def __init__(self, global_id: str, camera_id: str, timestamp: datetime, thumbnail_path: str = ""):
        self.global_id = global_id
        self.entry_time = timestamp
        self.last_seen_time = timestamp
        self.exit_time: Optional[datetime] = None
        self.entry_camera = camera_id
        self.last_camera = camera_id
        self.camera_path = [camera_id]
        self.status = "INSIDE"
        self.total_detections = 1
        self.thumbnail_path = thumbnail_path
        self.dwell_time_seconds = 0.0

    @property
    def dwell_time_str(self) -> str:
        seconds = int(self.dwell_time_seconds)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        return f"{mins:02d}m {secs:02d}s"


class JourneyManager:
    def __init__(self, db_path: str = str(config.DATABASE_PATH)):
        self.db_path = db_path
        self._init_database()
        
        self.active_visitors: Dict[str, VisitorRecord] = {}
        self.completed_visitors: Dict[str, VisitorRecord] = {}
        self.recent_events: List[Dict[str, Any]] = []
        
    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visitors (
                    global_id TEXT PRIMARY KEY,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    dwell_time_seconds REAL,
                    dwell_time_str TEXT,
                    status TEXT NOT NULL,
                    entry_camera TEXT NOT NULL,
                    last_camera TEXT NOT NULL,
                    camera_sequence TEXT NOT NULL,
                    total_detections INTEGER NOT NULL,
                    thumbnail_path TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    bbox TEXT,
                    FOREIGN KEY (global_id) REFERENCES visitors(global_id)
                )
            ''')
            conn.commit()

    def update_detection(self, global_id: str, camera_id: str, timestamp: datetime,
                         bbox: Optional[List[int]] = None, thumbnail_path: str = "",
                         is_new: bool = False):
        now = timestamp
        
        if global_id not in self.active_visitors and global_id not in self.completed_visitors:
            record = VisitorRecord(global_id, camera_id, now, thumbnail_path)
            self.active_visitors[global_id] = record
            
            event_type = "ENTRY"
            msg = f"{global_id} ENTERED at {camera_id} [{now.strftime('%H:%M:%S')}]"
            self.recent_events.append({"time": now.strftime('%H:%M:%S'), "text": msg, "type": "ENTRY", "id": global_id})
            
            self._db_insert_visitor(record)
            self._db_insert_event(global_id, camera_id, now, event_type, bbox)
            
        elif global_id in self.active_visitors:
            record = self.active_visitors[global_id]
            record.last_seen_time = now
            record.total_detections += 1
            record.dwell_time_seconds = max(0.0, (now - record.entry_time).total_seconds())
            
            if thumbnail_path and not record.thumbnail_path:
                record.thumbnail_path = thumbnail_path
                
            if record.last_camera != camera_id:
                record.camera_path.append(camera_id)
                record.last_camera = camera_id
                event_type = "TRANSITION"
                msg = f"{global_id} moved to {camera_id} [{now.strftime('%H:%M:%S')}]"
                self.recent_events.append({"time": now.strftime('%H:%M:%S'), "text": msg, "type": "TRANSITION", "id": global_id})
                self._db_insert_event(global_id, camera_id, now, event_type, bbox)

        if len(self.recent_events) > 30:
            self.recent_events.pop(0)

    def mark_exit(self, global_id: str, exit_time: datetime, reason: str = "Exit Camera"):
        if global_id in self.active_visitors:
            record = self.active_visitors.pop(global_id)
            record.exit_time = exit_time
            record.dwell_time_seconds = max(0.0, (exit_time - record.entry_time).total_seconds())
            record.status = "EXITED"
            self.completed_visitors[global_id] = record
            
            msg = f"{global_id} EXITED premises [{record.dwell_time_str}] ({reason})"
            self.recent_events.append({
                "time": exit_time.strftime('%H:%M:%S'),
                "text": msg,
                "type": "EXIT",
                "id": global_id
            })
            
            self._db_update_visitor(record)
            self._db_insert_event(global_id, record.last_camera, exit_time, "EXIT", None)

    def check_timeouts(self, current_time: datetime, timeout_seconds: float = config.EXIT_TIMEOUT_SECONDS):
        to_exit = []
        for gid, record in list(self.active_visitors.items()):
            cam_info = config.DEFAULT_CAMERAS.get(record.last_camera, {})
            is_exit_cam = cam_info.get("is_exit", False)
            elapsed = (current_time - record.last_seen_time).total_seconds()
            
            thresh = 2.5 if is_exit_cam else timeout_seconds
            if elapsed > thresh:
                reason = "Exit Gate Departure" if is_exit_cam else "Departed Camera View"
                # The true exit time is when the visitor was last seen leaving the frame
                to_exit.append((gid, record.last_seen_time, reason))
                
        for gid, exit_time, reason in to_exit:
            self.mark_exit(gid, exit_time, reason=reason)

    def merge_short_overlapping_visitors(self, min_detections: int = 150):
        """
        Post-processing step: merge brief visitors that have overlapping time windows.
        These are typically duplicate BoT-SORT tracks of the same brief background person.
        
        Visitors with >= min_detections are 'established' and never merged.
        Visitors with < min_detections that overlap in time are merged (shorter into longer).
        
        Example: Walker detected as Person-004 (64 detections, 271-357) and Person-005
        (58 detections, 318-393) both qualify and overlap → merged into one.
        Man (683) and Woman (743) are both >> 150 → always protected.
        """
        all_gids = list(self.active_visitors.keys()) + list(self.completed_visitors.keys())
        absorbed = {}  # gid_to_drop -> gid_to_keep

        for i, gid1 in enumerate(all_gids):
            if gid1 in absorbed:
                continue
            v1 = (self.active_visitors.get(gid1) or self.completed_visitors.get(gid1))
            if v1 is None or v1.total_detections >= min_detections:
                continue  # Protect established visitors

            for gid2 in all_gids[i + 1:]:
                if gid2 in absorbed:
                    continue
                v2 = (self.active_visitors.get(gid2) or self.completed_visitors.get(gid2))
                if v2 is None or v2.total_detections >= min_detections:
                    continue

                # Check temporal overlap (or close gap < 3 seconds)
                t1s, t1e = v1.entry_time, v1.last_seen_time
                t2s, t2e = v2.entry_time, v2.last_seen_time
                overlap = t1s <= t2e and t2s <= t1e
                gap_sec = min(abs((t2s - t1e).total_seconds()), abs((t1s - t2e).total_seconds()))
                if not overlap and gap_sec >= 3.0:
                    continue

                # Merge: absorb visitor with fewer detections into the other
                keep, drop = (gid1, gid2) if v1.total_detections >= v2.total_detections else (gid2, gid1)
                absorbed[drop] = keep

        for drop_gid, keep_gid in absorbed.items():
            if drop_gid in self.active_visitors:
                drop_rec = self.active_visitors.pop(drop_gid)
            elif drop_gid in self.completed_visitors:
                drop_rec = self.completed_visitors.pop(drop_gid)
            else:
                continue

            keep_rec = self.active_visitors.get(keep_gid) or self.completed_visitors.get(keep_gid)
            if keep_rec:
                keep_rec.total_detections += drop_rec.total_detections
                if drop_rec.entry_time < keep_rec.entry_time:
                    keep_rec.entry_time = drop_rec.entry_time
                if drop_rec.last_seen_time > keep_rec.last_seen_time:
                    keep_rec.last_seen_time = drop_rec.last_seen_time
                    keep_rec.dwell_time_seconds = max(0.0,
                        (keep_rec.last_seen_time - keep_rec.entry_time).total_seconds())

            # Also update SQLite records if drop_gid was previously inserted
            try:
                with self._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM visitors WHERE global_id = ?", (drop_gid,))
                    cur.execute("UPDATE camera_events SET global_id = ? WHERE global_id = ?", (keep_gid, drop_gid))
                    conn.commit()
            except Exception:
                pass

            # Log merge event
            msg = f"[Merge] {drop_gid} → {keep_gid} (duplicate brief visitor collapsed)"
            self.recent_events.append({"time": "", "text": msg, "type": "MERGE", "id": keep_gid})

    def flush_all_to_db(self, session_end_time: Optional[datetime] = None):
        """
        Flushes final stats for all visitors to SQLite on session finish.
        Merges any duplicate short-lived visitor fragments first,
        then marks departing individuals as EXITED with their last seen timestamp.
        """
        # Collapse brief overlapping duplicate tracks before final database write
        self.merge_short_overlapping_visitors(min_detections=150)

        now = session_end_time or datetime.now()
        for gid, record in list(self.active_visitors.items()):
            elapsed = (now - record.last_seen_time).total_seconds()
            if elapsed > 1.5:
                self.mark_exit(gid, record.last_seen_time, reason="Departed Camera View")
            else:
                self.mark_exit(gid, now, reason="Session Concluded")
                
        for record in self.completed_visitors.values():
            self._db_update_visitor(record)

    def _db_insert_visitor(self, record: VisitorRecord):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO visitors 
                (global_id, entry_time, exit_time, dwell_time_seconds, dwell_time_str,
                 status, entry_camera, last_camera, camera_sequence, total_detections, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.global_id,
                record.entry_time.isoformat(),
                record.exit_time.isoformat() if record.exit_time else None,
                record.dwell_time_seconds,
                record.dwell_time_str,
                record.status,
                record.entry_camera,
                record.last_camera,
                json.dumps(record.camera_path),
                record.total_detections,
                record.thumbnail_path
            ))
            conn.commit()

    def _db_update_visitor(self, record: VisitorRecord):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE visitors 
                SET exit_time = ?, dwell_time_seconds = ?, dwell_time_str = ?, 
                    status = ?, last_camera = ?, camera_sequence = ?, total_detections = ?,
                    thumbnail_path = ?
                WHERE global_id = ?
            ''', (
                record.exit_time.isoformat() if record.exit_time else None,
                record.dwell_time_seconds,
                record.dwell_time_str,
                record.status,
                record.last_camera,
                json.dumps(record.camera_path),
                record.total_detections,
                record.thumbnail_path,
                record.global_id
            ))
            conn.commit()

    def _db_insert_event(self, global_id: str, camera_id: str, timestamp: datetime,
                         event_type: str, bbox: Optional[List[int]]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO camera_events (global_id, camera_id, timestamp, event_type, bbox)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                global_id,
                camera_id,
                timestamp.isoformat(),
                event_type,
                json.dumps(bbox) if bbox else None
            ))
            conn.commit()

    def get_summary_metrics(self, current_time: Optional[datetime] = None,
                            live_camera_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        total_entered = len(self.active_visitors) + len(self.completed_visitors)
        currently_present = len(self.active_visitors)
        total_exited = len(self.completed_visitors)
        
        camera_counts = {}
        for cam_id in config.DEFAULT_CAMERAS.keys():
            camera_counts[cam_id] = 0
            
        if live_camera_counts:
            for cam_id, cnt in live_camera_counts.items():
                camera_counts[cam_id] = cnt
        else:
            for record in self.active_visitors.values():
                if record.last_camera in camera_counts:
                    camera_counts[record.last_camera] += 1
                else:
                    camera_counts[record.last_camera] = 1
                
        if self.completed_visitors:
            avg_dwell = sum(v.dwell_time_seconds for v in self.completed_visitors.values()) / len(self.completed_visitors)
            mins, secs = divmod(int(avg_dwell), 60)
            avg_dwell_str = f"{mins}m {secs}s"
        else:
            avg_dwell_str = "0m 00s"
            
        return {
            "total_entered": total_entered,
            "currently_present": currently_present,
            "total_exited": total_exited,
            "camera_counts": camera_counts,
            "avg_dwell_str": avg_dwell_str,
            "active_visitors": list(self.active_visitors.values()),
            "recent_events": self.recent_events[-6:]
        }