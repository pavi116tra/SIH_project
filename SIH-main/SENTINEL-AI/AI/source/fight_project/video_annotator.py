"""
video_annotator.py

Drop this into fight_project/. It takes:
  - the raw input video
  - per-frame person boxes (from your existing tracker.py / YOLO+ByteTrack)
  - per-window fight scores + time ranges (from your existing pipeline.py sliding window)

...and writes a new annotated .mp4 where frames inside a detected fight window get:
  - a RED rectangle around the interacting people
  - a "FIGHTING DETECTED" banner with the live threat score
and frames outside any fight window get a normal GREEN box per tracked person.

This is the piece that turns your side-panel timeline into an actual burned-in overlay
on the video itself.
"""

import cv2
import numpy as np


def frame_is_in_fight_window(frame_time, window_results, threshold):
    """
    window_results: list of dicts like
        {"start_time": 0.13, "end_time": 1.07, "score": 0.658}
        or {"start_sec": 0.13, "end_sec": 1.07, "combined_score": 0.658}
    """
    best_score = 0.0
    hit = False
    for w in window_results:
        start_t = w.get("start_sec", w.get("start_time", 0.0))
        end_t = w.get("end_sec", w.get("end_time", 0.0))
        score_val = w.get("combined_score", w.get("score", 0.0))
        if start_t <= frame_time <= end_t and score_val >= threshold:
            hit = True
            best_score = max(best_score, score_val)
    return hit, best_score



def enclosing_box(boxes, pad=15):
    """Given multiple [x1,y1,x2,y2] person boxes, return one box that encloses
    the closest interacting pair (or all boxes if you prefer a simpler version)."""
    if not boxes:
        return None
    boxes = np.array(boxes)
    x1 = max(int(boxes[:, 0].min()) - pad, 0)
    y1 = max(int(boxes[:, 1].min()) - pad, 0)
    x2 = int(boxes[:, 2].max()) + pad
    y2 = int(boxes[:, 3].max()) + pad
    return [x1, y1, x2, y2]


def closest_pair_box(boxes, pad=15):
    """More precise than enclosing_box: finds the two closest people (the ones
    likely actually interacting) instead of boxing in everyone in frame."""
    if len(boxes) < 2:
        return enclosing_box(boxes, pad)

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    best_pair, best_dist = None, float("inf")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)

    i, j = best_pair
    pair_boxes = [boxes[i], boxes[j]]
    return enclosing_box(pair_boxes, pad)


def draw_fight_alert(frame, box, score):
    """Red box + banner, drawn ON the frame (mutates and returns it)."""
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # BGR red

    label = f"!!! FIGHTING DETECTED !!!  {score * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    # solid banner background at the top of the frame so text is always legible
    cv2.rectangle(frame, (0, 0), (max(tw + 20, frame.shape[1]), th + 20), (0, 0, 180), -1)
    cv2.putText(frame, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_normal_boxes(frame, boxes, track_ids=None):
    """Green boxes for tracked people when no fight is active in this frame."""
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if track_ids:
            cv2.putText(frame, f"ID {track_ids[idx]}", (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return frame


def annotate_video(input_path, output_path, per_frame_boxes, window_results,
                    threshold, fps=None):
    """
    per_frame_boxes: list, one entry per frame, each entry is a list of
        [x1, y1, x2, y2] person boxes for that frame (from tracker.py output)
    window_results: list of {"start_time","end_time","score"} from your
        sliding-window fight classifier
    """
    cap = cv2.VideoCapture(input_path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_idx / fps
        boxes = per_frame_boxes[frame_idx] if frame_idx < len(per_frame_boxes) else []

        is_fight, score = frame_is_in_fight_window(frame_time, window_results, threshold)

        if is_fight:
            box = closest_pair_box(boxes) if boxes else None
            frame = draw_fight_alert(frame, box, score)
        else:
            frame = draw_normal_boxes(frame, boxes)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
