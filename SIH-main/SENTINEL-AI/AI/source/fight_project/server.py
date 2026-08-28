import os
import shutil
import uuid
import asyncio
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import DeepMultimodalThreatPipeline
from alert_system import AlertSystem

app = FastAPI(
    title="Sentinel Deep Frame-by-Frame Threat & Weapon Detection API",
    description="Multimodal API incorporating X3D-S + ResNet18-BiLSTM, Weapon & Threat Object Detector (Knives/Firearms), Muzzle Flash Anomaly Detection, Optical Flow, and Rule-Based Sensor Fusion.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "alerts", "snapshots")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/alerts/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")

try:
    pipeline = DeepMultimodalThreatPipeline(fight_threshold=0.60, camera_id="MAIN_CAM")
    alert_system = AlertSystem()
except Exception as e:
    print(f"Error initializing Deep Threat Pipeline: {e}")
    pipeline = None
    alert_system = AlertSystem()

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Sentinel Deep Threat & Weapon Detection API is Running!</h1>"

@app.post("/api/upload-video")
async def analyze_video_upload(file: UploadFile = File(...), threshold: float = Form(0.60)):
    """
    Deep Frame-by-Frame Video Analyzer:
    - Runs 16-frame sliding windows (stride=4 frames) across full uploaded video.
    - Evaluates:
      1. Weapon & Handheld Threat Objects (Knives, Guns, Sharp/Blunt Items)
      2. Muzzle Flash Anomaly & Visual Burst Spikes
      3. Farneback Optical Flow Motion Magnitude (Kicking, Punching, Throwing, Hitting)
      4. Person Proximity & Bounding Box Overlap
      5. X3D-S + ResNet18-BiLSTM Ensemble Confidence
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Fight Detection Pipeline failed to load.")

    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    raw_filename = f"upload_{file_id}{ext}"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)

    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video file: {e}")

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        os.remove(raw_path)
        raise HTTPException(status_code=400, detail=f"Corrupted video file: Unable to open '{file.filename}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        all_frames.append(frame)

    cap.release()

    if len(all_frames) == 0:
        os.remove(raw_path)
        raise HTTPException(status_code=400, detail=f"No readable video frames found in '{file.filename}'.")

    # 1. Run Deep Frame-by-Frame & Window Multimodal Inference
    pipeline.fight_threshold = threshold
    try:
        analysis_result = pipeline.analyze_full_video_deep(all_frames, fps=fps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep video threat analysis failed: {e}")

    overall_threat = analysis_result["overall_threat_level"]
    max_confidence = analysis_result["max_confidence"]
    threat_spans = analysis_result["threat_spans"]
    window_evaluations = analysis_result["window_evaluations"]
    frame_threat_status = analysis_result["frame_threat_status"]
    all_weapons_found = analysis_result["all_weapons_found"]

    out_filename = f"annotated_{file_id}.mp4"
    out_path = os.path.join(UPLOADS_DIR, out_filename)

    height, width = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # 2. Frame-by-Frame Render with Weapon Bounding Boxes & HUD Header
    results = pipeline.yolo(all_frames, classes=[0], verbose=False)

    for idx, frame in enumerate(all_frames):
        ann_frame = frame.copy()
        t_level = frame_threat_status[idx] if idx < len(frame_threat_status) else "NORMAL"

        p_boxes = []
        if idx < len(results) and results[idx].boxes is not None:
            for b in results[idx].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                p_boxes.append((x1, y1, x2, y2))

        timestamp_sec = round(idx / fps, 2)

        # Detect weapons in current frame for bounding box rendering
        weapons_curr = pipeline.weapon_detector.detect_weapons(frame)

        if t_level == "CRITICAL WEAPON THREAT":
            # Purple / Red Header
            cv2.rectangle(ann_frame, (0, 0), (width, 45), (128, 0, 128), -1)
            cv2.putText(ann_frame, f"CRITICAL THREAT: WEAPON / FLASH DETECTED at {timestamp_sec}s!",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in p_boxes:
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            for w_item in weapons_curr:
                wx1, wy1, wx2, wy2 = w_item["bbox"]
                cv2.rectangle(ann_frame, (wx1, wy1), (wx2, wy2), (255, 0, 255), 3)
                cv2.putText(ann_frame, f"WEAPON: {w_item['class_name']}", (wx1, max(15, wy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif t_level == "PHYSICAL ALTERCATION DETECTED":
            # Red Header
            cv2.rectangle(ann_frame, (0, 0), (width, 45), (0, 0, 220), -1)
            cv2.putText(ann_frame, f"!!! PHYSICAL ALTERCATION DETECTED at {timestamp_sec}s !!!",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in p_boxes:
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(ann_frame, "FIGHTER", (x1 + 5, max(15, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif t_level == "SUSPICIOUS ALTERCATION":
            # Orange Header
            cv2.rectangle(ann_frame, (0, 0), (width, 35), (0, 165, 255), -1)
            cv2.putText(ann_frame, f"SUSPICIOUS MOTION / ALTERCATION at {timestamp_sec}s",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in p_boxes:
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        else:
            # Green Header
            cv2.rectangle(ann_frame, (0, 0), (width, 35), (40, 40, 40), -1)
            cv2.putText(ann_frame, f"Status: NORMAL | Time: {timestamp_sec}s",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for (x1, y1, x2, y2) in p_boxes:
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(ann_frame)

    writer.release()

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Trigger Alert Log
    alerts = []
    if overall_threat in ["CRITICAL WEAPON THREAT", "PHYSICAL ALTERCATION DETECTED"]:
        alert_rec = alert_system.trigger_alert(
            camera_id="DEEP_THREAT_ANALYZER",
            timestamp_str=timestamp_str,
            confidence=max_confidence,
            person_ids=[1, 2],
            frame=all_frames[len(all_frames) // 2]
        )
        if alert_rec:
            alerts.append(alert_rec)

    return {
        "status": "success",
        "filename": file.filename,
        "total_frames": len(all_frames),
        "video_duration_sec": round(len(all_frames) / fps, 2),
        "overall_threat_level": overall_threat,
        "max_confidence": round(max_confidence, 4),
        "timestamp": timestamp_str,
        "annotated_video_url": f"/uploads/{out_filename}",
        "threat_spans": threat_spans,
        "weapons_found": all_weapons_found,
        "window_evaluations": window_evaluations,
        "alerts": alerts
    }

def generate_mjpeg_stream(rtsp_url: str):
    if pipeline is None:
        raise RuntimeError("Pipeline uninitialized.")

    source = rtsp_url
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to connect to stream source: {rtsp_url}")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            ann_frame, _, _, _ = pipeline.process_frame(frame)
            ret_enc, buffer = cv2.imencode('.jpg', ann_frame)
            if not ret_enc:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/api/live-stream")
def live_stream(rtsp_url: str = "0"):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Fight Detection Pipeline uninitialized.")
    return StreamingResponse(
        generate_mjpeg_stream(rtsp_url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    alerts = alert_system.get_recent_alerts(limit=limit)
    return {"status": "success", "count": len(alerts), "alerts": alerts}

if __name__ == "__main__":
    import uvicorn
    print("Starting Sentinel Deep Threat & Weapon Server on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
