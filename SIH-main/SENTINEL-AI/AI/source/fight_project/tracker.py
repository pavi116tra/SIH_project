import math
import numpy as np
import cv2
from ultralytics import YOLO

class PersonTracker:
    """
    Tracks human targets across video frames using YOLOv8 and ByteTrack algorithm.
    Extracts person bounding boxes and persistent track IDs.
    """
    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.4):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def track_frame(self, frame):
        """
        Runs YOLO person tracking on a single BGR image frame.
        Returns list of tracked person dicts:
        [{'id': track_id, 'bbox': [x1, y1, x2, y2], 'center': (cx, cy), 'conf': conf}]
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0], # Person class
            conf=self.conf_thresh,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_persons = []
        if not results or len(results) == 0:
            return tracked_persons

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracked_persons

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tracked_persons.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "conf": conf
            })

        return tracked_persons

def find_interacting_persons(tracked_persons, distance_threshold=200):
    """
    Identifies pairs of tracked persons whose bounding boxes or centers are closest together.
    Returns list of track IDs of interacting individuals.
    """
    if len(tracked_persons) < 2:
        return [p["id"] for p in tracked_persons if p["id"] != -1]

    interacting_ids = set()

    for i in range(len(tracked_persons)):
        for j in range(i + 1, len(tracked_persons)):
            p1 = tracked_persons[i]
            p2 = tracked_persons[j]

            c1 = p1["center"]
            c2 = p2["center"]

            # Euclidean distance between bounding box centers
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

            # Check IoU/Overlapping bounding boxes
            b1 = p1["bbox"]
            b2 = p2["bbox"]
            
            x_left = max(b1[0], b2[0])
            y_top = max(b1[1], b2[1])
            x_right = min(b1[2], b2[2])
            y_bottom = min(b1[3], b2[3])

            overlap = max(0, x_right - x_left) * max(0, y_bottom - y_top)

            if dist < distance_threshold or overlap > 0:
                interacting_ids.add(p1["id"])
                interacting_ids.add(p2["id"])

    # Fallback: if distance threshold misses, select the two closest individuals
    if not interacting_ids:
        min_dist = float('inf')
        pair = (tracked_persons[0]["id"], tracked_persons[1]["id"])
        for i in range(len(tracked_persons)):
            for j in range(i + 1, len(tracked_persons)):
                c1 = tracked_persons[i]["center"]
                c2 = tracked_persons[j]["center"]
                dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    pair = (tracked_persons[i]["id"], tracked_persons[j]["id"])
        interacting_ids.update(pair)

    return list(interacting_ids)

if __name__ == "__main__":
    print("PersonTracker initialized successfully.")
