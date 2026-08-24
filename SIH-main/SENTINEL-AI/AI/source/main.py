"""
Standalone Real-Time Human Detection & Multi-Object Tracking Application.
Executes live video stream processing from Webcam, RTSP stream, or Video file.
"""

import argparse
import cv2
import os
import sys

# Ensure local modules are in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.human_tracker import HumanTrackerEngine
from visualization.renderer import SurveillanceRenderer


def run_tracking_system(source=0, config_path=None, show_window=True, output_path=None):
    """
    Run human detection, multi-object tracking, and trajectory visualization.
    """
    engine = HumanTrackerEngine(config_path=config_path)
    renderer = SurveillanceRenderer()

    # Parse camera / video input source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[SENTINEL-AI] Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source '{source}'.")
        return False

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w_frame, h_frame))
        print(f"[SENTINEL-AI] Recording output to: {output_path}")

    print("[SENTINEL-AI] Tracking system active. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[SENTINEL-AI] End of video stream reached.")
                break

            # Process frame through human detection & tracker
            human_tracks, non_human_objects, stats = engine.process_frame(frame)

            # Render annotations over frame
            annotated_frame = renderer.draw_annotations(
                frame, human_tracks, non_human_objects, stats
            )

            if video_writer:
                video_writer.write(annotated_frame)

            if show_window:
                cv2.imshow("SENTINEL-AI - High-Accuracy Human Tracker", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\n[SENTINEL-AI] System interrupted by user.")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if show_window:
            cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("         FINAL SESSION ANALYTICS         ")
        print("=" * 50)
        print(f"Total Frames Processed : {engine.frame_count}")
        print(f"Unique Humans Observed : {len(engine.unique_human_ids)}")
        print(f"Average FPS            : {round(engine.last_fps, 1)}")
        print("=" * 50)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SENTINEL-AI Real-Time Human Detection & Tracking Platform"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (0), RTSP URL, or video file path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml configuration file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display window",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated MP4 video output",
    )
    args = parser.parse_args()

    run_tracking_system(
        source=args.source,
        config_path=args.config,
        show_window=not args.no_show,
        output_path=args.output,
    )
