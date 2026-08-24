"""
Multi-Camera Manager.
Manages CameraSource abstractions, config-driven camera setup, start/stop controls,
and cross-camera topology transition logs.
"""

import time
import os
import yaml
from camera.camera_source import create_camera_source


class CameraManager:
    """
    Manages active camera sources (webcam, video files, RTSP streams) and topology logs.
    """

    def __init__(self, config_path=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if config_path is None:
            config_path = os.path.join(self.base_dir, "config", "config.yaml")

        self.sources = {}  # {camera_id: CameraSource}
        self.transition_logs = []

        self._load_from_config(config_path)
        self.start_all_enabled()

    def _load_from_config(self, config_path):
        cams_cfg = []
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    cams_cfg = cfg.get("cameras", [])
            except Exception as e:
                print(f"[CameraManager] Error loading config {config_path}: {e}")

        # Default fallback if config empty: CAM01 Laptop Webcam
        if not cams_cfg:
            cams_cfg = [
                {
                    "id": "CAM01",
                    "name": "Laptop Webcam",
                    "source_type": "webcam",
                    "source": 0,
                    "enabled": True,
                }
            ]

        self.reset_callback = None

        for cdict in cams_cfg:
            cam_obj = create_camera_source(cdict)
            self.sources[cam_obj.camera_id] = cam_obj

    def set_reset_callback(self, callback):
        """Register reset hook for all camera sources."""
        self.reset_callback = callback
        for source in self.sources.values():
            source.on_reset_callback = callback

    def start_camera(self, camera_id):
        """Start a specific camera stream."""
        if camera_id in self.sources:
            return self.sources[camera_id].start()
        return False

    def stop_camera(self, camera_id):
        """Stop a specific camera stream."""
        if camera_id in self.sources:
            self.sources[camera_id].stop()
            return True
        return False

    def pause_camera(self, camera_id):
        """Pause frame playback for a specific camera."""
        if camera_id in self.sources:
            self.sources[camera_id].pause()
            return True
        return False

    def resume_camera(self, camera_id):
        """Resume frame playback for a specific camera."""
        if camera_id in self.sources:
            self.sources[camera_id].resume()
            return True
        return False

    def restart_camera(self, camera_id):
        """Restart stream playback from beginning for a specific camera."""
        if camera_id in self.sources:
            self.sources[camera_id].restart()
            return True
        return False

    def start_all_enabled(self):
        """Start all enabled camera streams."""
        results = {}
        for cid, source in self.sources.items():
            if source.enabled:
                results[cid] = source.start()
        return results

    def read_frame(self, camera_id):
        """Read a frame from a specific camera ID."""
        if camera_id in self.sources:
            return self.sources[camera_id].read_frame()
        return False, None

    def get_active_cameras(self):
        """Return list of status dicts for all configured cameras."""
        return [s.get_status() for s in self.sources.values()]

    def log_camera_transition(self, global_id, from_cam, to_cam):
        """Log a cross-camera transition event."""
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_id": global_id,
            "from_camera": from_cam,
            "to_camera": to_cam,
        }
        self.transition_logs.append(event)
        print(
            f"[CameraManager] CROSS-CAMERA TRANSITION DETECTED! {global_id}: {from_cam} -> {to_cam}"
        )
