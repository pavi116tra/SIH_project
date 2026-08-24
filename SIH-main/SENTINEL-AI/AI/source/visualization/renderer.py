"""
Real-time Video Visualization Renderer.
Renders bounding boxes, persistent track IDs, class badges, motion trajectory trails,
and an executive surveillance HUD panel over video frames.
"""

import cv2
import numpy as np


class SurveillanceRenderer:
    """
    Renders visual annotations and real-time surveillance statistics on frames.
    """

    def __init__(self):
        # Color schemes (BGR)
        self.color_person = (0, 255, 0)        # Vibrant Green for Humans
        self.color_non_person = (255, 165, 0)  # Cyan/Orange for Non-Humans
        self.color_trail = (0, 215, 255)       # Gold/Yellow for Trajectories
        self.color_hud_bg = (20, 20, 20)       # Dark Glassmorphic HUD background
        self.color_text = (255, 255, 255)      # Pure White text

    def draw_annotations(self, frame, human_tracks, non_human_objects, stats):
        """
        Draw all bounding boxes, track IDs, motion trajectories, and HUD over the frame.
        """
        canvas = frame.copy()
        h_frame, w_frame = canvas.shape[:2]

        # 1. Draw Motion Trajectory Trails for Active Humans
        for track in human_tracks:
            if len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=False,
                    color=self.color_trail,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                # Draw small circle at current position
                last_pt = (int(track.history[-1][0]), int(track.history[-1][1]))
                cv2.circle(
                    canvas,
                    last_pt,
                    4,
                    self.color_trail,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        # 2. Draw Non-Human Objects (Vehicles, Birds, Animals, etc.)
        for obj in non_human_objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            score_pct = int(obj["score"] * 100)
            label = f"{obj['class_name']} | {score_pct}%"

            # Draw dashed/dotted or standard rectangle
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), self.color_non_person, 2, cv2.LINE_AA
            )

            # Label badge
            (w_lbl, h_lbl), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                canvas,
                (x1, max(0, y1 - h_lbl - 6)),
                (x1 + w_lbl + 8, y1),
                self.color_non_person,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Humans with Persistent Tracking IDs
        for track in human_tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            local_id = track.track_id
            score_pct = int(getattr(track, "score", 0.95) * 100)
            gid = getattr(track, "global_id", None)
            cam_id = getattr(track, "camera_id", "C1")
            gid_str = f"GLOBAL ID: {gid}" if gid else "Resolving..."
            label = f"{gid_str} | {cam_id}-Track #{local_id} | {score_pct}%"

            # Draw thick vibrant box
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), self.color_person, 2, cv2.LINE_AA
            )

            # Top label box
            (w_lbl, h_lbl), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(
                canvas,
                (x1, max(0, y1 - h_lbl - 8)),
                (x1 + w_lbl + 10, y1),
                self.color_person,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 4. Draw Executive Surveillance HUD Badge (Sleek, Compact, Non-Obstructive)
        hud_str = f"Visible Humans: {stats.get('visible_human_count', 0)} | Unique: {stats.get('unique_human_count', 0)}"
        (w_hud, h_hud), _ = cv2.getTextSize(hud_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (10 + w_hud + 16, 36), self.color_hud_bg, -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (10 + w_hud + 16, 36), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            hud_str,
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return canvas
