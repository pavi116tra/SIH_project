import os
import time
import json
import sqlite3
from datetime import datetime
import cv2
import requests

class AlertSystem:
    """
    Manages fight alert notifications, snapshot persistence, SQLite logging,
    webhook broadcasting, and camera cooldown handling.
    """
    def __init__(self, db_path="alerts.db", snapshot_dir="alerts/snapshots", cooldown_seconds=5.0, webhook_url=None):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self.cooldown_seconds = cooldown_seconds
        self.webhook_url = webhook_url
        self.last_alert_time = {} # camera_id -> float timestamp

        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                person_ids TEXT NOT NULL,
                snapshot_path TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def trigger_alert(self, camera_id, timestamp_str, confidence, person_ids, frame):
        """
        Triggers a fight alert if cooldown window has passed.
        Returns dict with alert record if triggered, or None if suppressed by cooldown.
        """
        current_time = time.time()
        last_time = self.last_alert_time.get(camera_id, 0.0)

        # 1. Cooldown Check
        if (current_time - last_time) < self.cooldown_seconds:
            # Alert suppressed due to active cooldown window
            return None

        self.last_alert_time[camera_id] = current_time

        # 2. Save Evidence Snapshot Image
        time_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        snapshot_filename = f"alert_{camera_id}_{time_slug}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        
        if frame is not None:
            # Draw alert banner on snapshot image for audit evidence
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 200), -1)
            cv2.putText(frame, f"WARNING: FIGHT DETECTED | Conf: {confidence*100:.1f}% | Cam: {camera_id}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(snapshot_path, frame)

        person_ids_str = json.dumps(person_ids)

        # 3. Log to SQLite Database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (camera_id, timestamp, confidence, person_ids, snapshot_path)
            VALUES (?, ?, ?, ?, ?)
        """, (str(camera_id), timestamp_str, float(confidence), person_ids_str, snapshot_path))
        conn.commit()
        conn.close()

        alert_record = {
            "camera_id": camera_id,
            "timestamp": timestamp_str,
            "confidence": round(float(confidence), 4),
            "person_ids": person_ids,
            "snapshot_path": snapshot_path
        }

        # 4. Log to JSON backup
        json_log_path = "alerts.json"
        existing_alerts = []
        if os.path.exists(json_log_path):
            try:
                with open(json_log_path, "r") as f:
                    existing_alerts = json.load(f)
            except Exception:
                existing_alerts = []
        existing_alerts.append(alert_record)
        with open(json_log_path, "w") as f:
            json.dump(existing_alerts, f, indent=4)

        # 5. POST to Webhook if configured
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=alert_record, timeout=2.0)
            except Exception as e:
                print(f"Webhook dispatch failed: {e}")

        print(f"\n[ALERT TRIGGERED] Camera: {camera_id} | Time: {timestamp_str} | Conf: {confidence*100:.1f}% | Persons: {person_ids}")
        return alert_record

    def get_recent_alerts(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        alerts = [dict(row) for row in rows]
        conn.close()
        return alerts

if __name__ == "__main__":
    alert_sys = AlertSystem()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    alert_sys.trigger_alert("CAM_01", datetime.now().isoformat(), 0.94, [1, 2], dummy_frame)
    print("Alert System initialized and tested successfully.")
