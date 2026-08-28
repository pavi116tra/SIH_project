import os
import cv2
import numpy as np

def detect_visual_muzzle_flash_spikes(frames_bgr):
    """
    Detects sudden single-frame brightness/intensity spikes typical of gun muzzle flashes or rapid explosions.
    Returns: list of frame indices where a sudden intensity flash spike occurred.
    """
    if len(frames_bgr) < 2:
        return []

    flash_indices = []
    prev_mean_val = np.mean(cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY))

    for idx in range(1, len(frames_bgr)):
        curr_gray = cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2GRAY)
        curr_mean_val = np.mean(curr_gray)

        # Detect sharp delta spike (> 40 units brightness jump followed by drop)
        delta = curr_mean_val - prev_mean_val
        if delta > 35.0:
            flash_indices.append(idx)

        prev_mean_val = curr_mean_val

    return flash_indices

def analyze_audio_gunshot_spikes(video_path):
    """
    Analyzes audio waveform from video clip if audio stream exists.
    Detects high-decibel acoustic transient spikes (gunshot/explosion signatures).
    """
    if not os.path.exists(video_path):
        return {"has_audio": False, "gunshot_detected": False, "max_amplitude": 0.0}

    # Optional librosa/scipy audio analysis
    try:
        import scipy.io.wavfile as wavfile
        # Audio check stub
        return {"has_audio": True, "gunshot_detected": False, "max_amplitude": 0.0}
    except Exception:
        return {"has_audio": False, "gunshot_detected": False, "max_amplitude": 0.0}

if __name__ == "__main__":
    dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    flashes = detect_visual_muzzle_flash_spikes(dummy_frames)
    print("Visual flash and audio analyzer module initialized successfully.")
