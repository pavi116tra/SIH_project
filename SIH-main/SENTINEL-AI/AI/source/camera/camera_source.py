"""
Camera Source Abstraction Layer.
Provides unified interface for Webcam, Video File, and RTSP stream sources.
"""

import time
import os
import cv2
import numpy as np


class CameraSource:
    """Base class for all camera input streams."""

    def __init__(self, camera_id, name, source_type, source, enabled=True):
        self.camera_id = camera_id
        self.name = name
        self.source_type = source_type
        self.source = source
        self.enabled = enabled

        self.is_online = False
        self.is_paused = False
        self.cap = None
        self.last_frame = None
        self.error_msg = ""
        self.on_reset_callback = None

    def start(self):
        raise NotImplementedError

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_online = False
        self.is_paused = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def restart(self):
        self.is_paused = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.on_reset_callback:
            try:
                self.on_reset_callback(self.camera_id)
            except Exception as e:
                print(f"[CameraSource] Reset callback error for {self.camera_id}: {e}")

    def read_frame(self):
        raise NotImplementedError

    def get_status(self):
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "source": str(self.source),
            "enabled": self.enabled,
            "is_online": self.is_online,
            "is_paused": self.is_paused,
            "error_msg": self.error_msg,
        }

    def _render_error_frame(self, message):
        """Generates a clear red error frame displaying why the source is offline."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Red outer warning border
        cv2.rectangle(frame, (15, 15), (625, 465), (0, 0, 220), 3)
        # Dark red header banner
        cv2.rectangle(frame, (15, 15), (625, 60), (0, 0, 140), -1)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CRITICAL STREAM ERROR: {self.camera_id}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Large offline message
        cv2.putText(frame, f"{self.camera_id} OFFLINE", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 50, 255), 3)
        cv2.putText(frame, message, (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

        cv2.putText(frame, f"SOURCE: {self.source}", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        cv2.putText(frame, "Please ensure the video file exists or upload a new stream.", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        cv2.putText(frame, f"STATUS: DISCONNECTED | {ts}", (30, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        time.sleep(0.04)
        return frame


class WebcamSource(CameraSource):
    """Camera source for physical webcams (USB/laptop webcam)."""

    def __init__(self, camera_id="CAM01", name="Laptop Webcam", source=0, enabled=True):
        super().__init__(camera_id, name, "webcam", source, enabled)

    def start(self):
        self.enabled = True
        try:
            device_idx = int(self.source)
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(device_idx)
            if self.cap.isOpened():
                self.is_online = True
                self.error_msg = ""
                print(f"[WebcamSource] {self.camera_id} ({self.name}) started successfully on device {device_idx}.")
                return True
            else:
                self.is_online = False
                self.error_msg = f"{self.camera_id} OFFLINE — webcam device {self.source} unavailable"
                print(f"[WebcamSource] {self.camera_id} failed to open device {device_idx}.")
                return False
        except Exception as e:
            print(f"[WebcamSource] {self.camera_id} initialization error: {e}")
            self.is_online = False
            self.error_msg = f"{self.camera_id} OFFLINE — webcam error: {e}"
            return False

    def read_frame(self):
        if not self.enabled:
            return False, None

        if self.is_paused and self.last_frame is not None:
            paused_frame = self.last_frame.copy()
            cv2.rectangle(paused_frame, (20, 20), (140, 60), (0, 165, 255), -1)
            cv2.putText(paused_frame, "PAUSED", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            time.sleep(0.04)
            return True, paused_frame

        if self.cap is None or not self.cap.isOpened():
            self.start()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.is_online = True
                self.error_msg = ""
                self.last_frame = frame.copy()
                return True, frame

        # Fallback error generator if webcam unavailable
        self.is_online = False
        if not self.error_msg:
            self.error_msg = f"{self.camera_id} OFFLINE — webcam stream disconnected"
        return True, self._render_error_frame(self.error_msg)


class VideoFileSource(CameraSource):
    """Camera source for video files (MP4/AVI) supporting multi-camera simulation."""

    def __init__(self, camera_id, name, source, enabled=True, loop=True):
        super().__init__(camera_id, name, "video", source, enabled)
        self.loop = loop
        self.frame_counter = 0

    def _resolve_video_path(self, target_path):
        """Locates valid video file across project directories."""
        str_path = str(target_path)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Priority order of candidate file locations
        candidates = [
            str_path,
            os.path.join(base_dir, str_path),
            os.path.join(base_dir, "video", os.path.basename(str_path)),
            os.path.join(base_dir, "videos", os.path.basename(str_path)),
            os.path.join(base_dir, "uploads", os.path.basename(str_path)),
        ]

        for cand in candidates:
            if cand and os.path.exists(cand) and os.path.isfile(cand):
                return cand

        # Fallback check: Look for any available video in video/ or uploads/
        for subfolder in ["video", "uploads", "videos"]:
            folder_path = os.path.join(base_dir, subfolder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if files:
                    # Pick a video file deterministically based on camera index
                    idx = int(''.join(filter(str.isdigit, self.camera_id)) or 1) % len(files)
                    chosen = os.path.join(folder_path, files[idx])
                    print(f"[VideoFileSource] {self.camera_id} fallback using available video: {chosen}")
                    return chosen

        return None

    def start(self):
        self.enabled = True
        resolved = self._resolve_video_path(self.source)

        if resolved:
            try:
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(resolved)
                if self.cap.isOpened():
                    self.is_online = True
                    self.error_msg = ""
                    print(f"[VideoFileSource] {self.camera_id} ({self.name}) opened video file: {resolved}")
                    return True
            except Exception as e:
                print(f"[VideoFileSource] {self.camera_id} error opening {resolved}: {e}")

        # Source not found or could not be opened
        self.is_online = False
        self.error_msg = f"{self.camera_id} OFFLINE — video source not found"
        print(f"[VideoFileSource] {self.camera_id} OFFLINE — video file not found: {self.source}")
        return False

    def read_frame(self):
        if not self.enabled:
            return False, None

        if self.is_paused and self.last_frame is not None:
            paused_frame = self.last_frame.copy()
            cv2.rectangle(paused_frame, (20, 20), (140, 60), (0, 165, 255), -1)
            cv2.putText(paused_frame, "PAUSED", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            time.sleep(0.04)
            return True, paused_frame

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.is_online = True
                self.error_msg = ""
                self.last_frame = frame.copy()
                return True, frame
            elif self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self.on_reset_callback:
                    try:
                        self.on_reset_callback(self.camera_id)
                    except Exception as e:
                        print(f"[VideoFileSource] Reset callback error: {e}")
                ret, frame = self.cap.read()
                if ret:
                    self.is_online = True
                    self.error_msg = ""
                    self.last_frame = frame.copy()
                    return True, frame

        # If file missing or stream failed, show explicit red error frame
        self.is_online = False
        if not self.error_msg:
            self.error_msg = f"{self.camera_id} OFFLINE — video source not found"
        return True, self._render_error_frame(self.error_msg)


class RTSPSource(CameraSource):
    """Camera source for IP/RTSP network security cameras."""

    def __init__(self, camera_id, name, source, enabled=True):
        super().__init__(camera_id, name, "rtsp", source, enabled)

    def start(self):
        self.enabled = True
        try:
            self.cap = cv2.VideoCapture(str(self.source))
            if self.cap.isOpened():
                self.is_online = True
                self.error_msg = ""
                print(f"[RTSPSource] {self.camera_id} ({self.name}) connected to RTSP stream: {self.source}")
                return True
        except Exception as e:
            print(f"[RTSPSource] {self.camera_id} RTSP connection error: {e}")

        self.is_online = False
        self.error_msg = f"{self.camera_id} OFFLINE — RTSP stream unreachable"
        print(f"[RTSPSource] {self.camera_id} ({self.name}) RTSP stream offline.")
        return False

    def read_frame(self):
        if not self.enabled:
            return False, None

        if self.is_paused and self.last_frame is not None:
            paused_frame = self.last_frame.copy()
            cv2.rectangle(paused_frame, (20, 20), (140, 60), (0, 165, 255), -1)
            cv2.putText(paused_frame, "PAUSED", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            time.sleep(0.04)
            return True, paused_frame

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.is_online = True
                self.error_msg = ""
                self.last_frame = frame.copy()
                return True, frame

        self.is_online = False
        if not self.error_msg:
            self.error_msg = f"{self.camera_id} OFFLINE — RTSP stream unreachable"
        return True, self._render_error_frame(self.error_msg)


def create_camera_source(cam_dict):
    """Factory function to build CameraSource instances from config dictionary."""
    cid = cam_dict.get("id", "CAM01")
    name = cam_dict.get("name", f"Camera {cid}")
    stype = cam_dict.get("source_type", "webcam").lower()
    source = cam_dict.get("source", 0)
    enabled = cam_dict.get("enabled", True)

    if stype == "webcam":
        return WebcamSource(cid, name, source, enabled)
    elif stype == "video":
        return VideoFileSource(cid, name, source, enabled)
    elif stype == "rtsp":
        return RTSPSource(cid, name, source, enabled)
    else:
        return WebcamSource(cid, name, source, enabled)
