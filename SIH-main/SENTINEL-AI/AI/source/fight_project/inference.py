import os
import sys
import argparse
import cv2
from pipeline import FightDetectionPipeline

def run_video_inference(video_source, output_video_path=None, show_display=False, threshold=0.70):
    """
    Runs real-time fight detection pipeline on a video file or live stream (RTSP URL / Webcam index).
    """
    # Check if video source is integer (webcam index)
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    print(f"\nInitializing Fight Detection Pipeline...")
    print(f"  Input Source: {video_source}")
    print(f"  Confidence Threshold: {threshold*100:.1f}%\n")

    pipeline = FightDetectionPipeline(fight_threshold=threshold)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    writer = None
    if output_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving output annotated video to '{output_video_path}'")

    frame_idx = 0
    alerts_triggered = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            ann_frame, is_fight, conf, alert_info = pipeline.process_frame(frame)

            if alert_info:
                alerts_triggered += 1

            if writer:
                writer.write(ann_frame)

            if show_display:
                cv2.imshow("Sentinel Real-Time Fight Detection", ann_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User interrupted display.")
                    break

            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames | Active Status: {'FIGHT' if is_fight else 'NORMAL'} | Conf: {conf*100:.1f}%")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()

    print(f"\n--- Inference Complete ---")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Fight Alerts Triggered: {alerts_triggered}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Fight Detection Inference Engine")
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index / RTSP URL")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated MP4 video file")
    parser.add_argument("--display", action="store_true", help="Show GUI window preview")
    parser.add_argument("--threshold", type=float, default=0.70, help="Fight confidence threshold (0.0 to 1.0)")
    
    args = parser.parse_args()
    run_video_inference(args.source, args.output, args.display, args.threshold)
