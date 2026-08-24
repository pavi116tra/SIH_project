import os
import sys

# Ensure virtual environment packages are discoverable by python and IDE language servers
_venv_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "venv", "Lib", "site-packages"))
if os.path.exists(_venv_packages) and _venv_packages not in sys.path:
    sys.path.insert(0, _venv_packages)

from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import base64
import re
import logging
import numpy as np
import mediapipe as mp
import torch
import pickle
from datetime import datetime
import time
from pymongo import MongoClient  # type: ignore
from email.message import EmailMessage
import smtplib
import ssl
import threading
from queue import Queue
from twilio.rest import Client
from ultralytics import YOLO  # type: ignore
import easyocr  # type: ignore
from facenet_pytorch import InceptionResnetV1  # type: ignore

class FaceNet:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embeddings(self, images):
        if not images:
            return np.empty((0, 512))
        tensors = []
        for img in images:
            if img.shape[:2] != (160, 160):
                img = cv2.resize(img, (160, 160))
            tensor = (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) - 127.5) / 128.0
            tensors.append(tensor)
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch).cpu().numpy()
        return embs

task_queue = Queue()
vehicle_queue = Queue()

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Email Configuration
EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'your_email@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your_email_password')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER', 'receiver_email@gmail.com')

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'your_auth_token_here')    
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', 'your_twilio_phone_number')      
RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER', 'your_recipient_phone_number') 

# Connect to MongoDB with local fallback
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.server_info() # force connection check
    db = client["face_recognition"]
    collection = db["suspects"]
    vehicles_collection = db["vehicles"]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB not available, using local file-based database: {e}")
    class LocalFallbackCollection:
        def __init__(self, filepath):
            self.filepath = filepath

        def _load(self):
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return []
            return []

        def _save(self, data):
            try:
                with open(self.filepath, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Error saving local db: {e}")

        def insert_one(self, document):
            data = self._load()
            from bson import ObjectId
            document = document.copy()
            if "_id" not in document:
                document["_id"] = ObjectId()
            data.append(document)
            self._save(data)
            return document

        def find(self, filter=None):
            return self._load()

    collection = LocalFallbackCollection(os.path.join(os.path.dirname(os.path.abspath(__file__)), "suspects_db.pkl"))
    vehicles_collection = LocalFallbackCollection(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicles_db.pkl"))


# Initialize models
embedder = FaceNet()
class CascadeFaceDetection:
    def __init__(self, *args, **kwargs):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    class Detection:
        def __init__(self, x, y, w, h, img_w, img_h):
            class RelativeBB:
                def __init__(self, x, y, w, h, img_w, img_h):
                    self.xmin = x / img_w
                    self.ymin = y / img_h
                    self.width = w / img_w
                    self.height = h / img_h
            class LocationData:
                def __init__(self, x, y, w, h, img_w, img_h):
                    self.relative_bounding_box = RelativeBB(x, y, w, h, img_w, img_h)
            self.location_data = LocationData(x, y, w, h, img_w, img_h)

    class Results:
        def __init__(self, detections):
            self.detections = detections

    def process(self, img_rgb):
        if img_rgb is None or img_rgb.size == 0:
            return self.Results([])
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        detections = [self.Detection(fx, fy, fw, fh, w, h) for (fx, fy, fw, fh) in faces]
        return self.Results(detections)

    def close(self):
        pass

class FaceDetectorWrapper:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
        else:
            self._detector = CascadeFaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence)

    def process(self, img_rgb):
        return self._detector.process(img_rgb)

    def close(self):
        if hasattr(self._detector, 'close'):
            self._detector.close()

class MPFaceDetectionFallback:
    FaceDetection = FaceDetectorWrapper

mp_face_detection = MPFaceDetectionFallback
face_detector = FaceDetectorWrapper(model_selection=1, min_detection_confidence=0.5)

from tracking.human_tracker import HumanTrackerEngine  # type: ignore
from visualization.renderer import SurveillanceRenderer  # type: ignore
from camera.camera_manager import CameraManager  # type: ignore

human_tracker_engine = HumanTrackerEngine()
surveillance_renderer = SurveillanceRenderer()
camera_manager = CameraManager()
camera_manager.set_reset_callback(human_tracker_engine.reset_camera_tracker)

current_tracking_stats = {
    "frame_id": 0,
    "fps": 0.0,
    "latency_ms": 0.0,
    "visible_human_count": 0,
    "unique_human_count": 0,
    "non_human_count": 0,
    "system_status": "ONLINE - ACTIVE TRACKING"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} for AI models.")

vehicle_model = YOLO("yolov8n.pt")
vehicle_model.to(device)
# Suppress EasyOCR verbosity
logging.getLogger("easyocr").setLevel(logging.ERROR)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# --- Find the best available plate detector model across all training runs ---
def find_best_plate_model():
    """Search all training run directories for the best available plate detector weights."""
    candidates = [
        "model/best.pt",  # manually placed model
        "runs/detect/license_plate_detector/weights/best.pt",
    ]
    # Also search numbered runs (license_plate_detector2, 3, 4, ...)
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    if os.path.isdir(runs_dir):
        for entry in sorted(os.listdir(runs_dir), reverse=True):
            # Prefer higher-numbered runs (trained later = better)
            candidate = os.path.join(runs_dir, entry, "weights", "best.pt")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                candidates.insert(0, candidate)
    
    for path in candidates:
        abs_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            return abs_path
    return None

plate_model_path = find_best_plate_model()
if plate_model_path:
    print(f"Loading Custom Trained YOLO ALPR model: {plate_model_path}")
    plate_detector = YOLO(plate_model_path)
    plate_detector.to(device)
else:
    print("Custom Plate Detector not found. Using Haar Cascade fallback.")
    plate_detector = None
    
# Plate detection cascade (Fallback)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Characters allowed on license plates (include separators like - and .)
PLATE_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
PLATE_ALLOWLIST_STRICT = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def clean_plate_text(raw_text):
    """Clean OCR output: keep only plate-valid characters, normalize separators."""
    text = raw_text.strip().upper()
    # Replace common OCR mis-reads for separators
    text = text.replace('|', '1').replace('\\', '1').replace('(', '').replace(')', '')
    text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    # Keep only valid plate characters
    cleaned = ''.join(c for c in text if c.isalnum() or c in '-.')
    # Remove leading/trailing separators
    cleaned = cleaned.strip('-.')
    return cleaned

def apply_ocr_corrections(text):
    """Apply post-processing corrections for known OCR confusions on license plate fonts.
    These are systematic errors where EasyOCR consistently misreads certain characters."""
    if not text:
        return text
    
    corrected = text
    
    # --- Fix merged characters ---
    # "51" often merges into "6" when characters are close together
    # If we see a plate starting with "6" followed by a letter, it's likely "51"
    # Vietnamese plates: ##X-###.## where ## is province code (2 digits), X is letter
    import re as _re
    
    # Pattern: plate starts with single digit + letter (like "6F" or "6E") 
    # but should be two digits + letter (like "51F", "51E")
    # Common province codes: 51 (Ho Chi Minh), 30 (Ha Noi), 43 (Da Nang), etc.
    match = _re.match(r'^(\d)([A-Z])', corrected)
    if match:
        single_digit = match.group(1)
        letter = match.group(2)
        rest = corrected[len(match.group(0)):]
        
        # Map of commonly merged digit pairs
        # When "5" and "1" are close, OCR reads "6"
        # When "3" and "1" are close, OCR reads "4" or "31"
        # When "6" and "1" are close, OCR reads "6" (absorbs the 1)
        merge_corrections = {
            '6': '51',  # 5+1 merged (most common for Vietnamese plates)
            '8': '51',  # 5+1 merged (alternative misread)
            '4': '41',  # 4+1 merged
            '9': '91',  # 9+1 merged (or could be actual 9)
        }
        
        if single_digit in merge_corrections:
            # Check if corrected version gives a known Vietnamese province code
            expanded = merge_corrections[single_digit]
            # Known Vietnamese province codes that start with these digits
            known_provinces = [
                '11','12','14','15','16','17','18','19','20','21','22','23','24','25','26','27',
                '29','30','31','32','33','34','35','36','37','38','39','40','41','43','47','48',
                '49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64',
                '65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80',
                '81','82','83','84','85','86','88','89','90','92','93','94','95','97','98','99'
            ]
            if expanded in known_provinces:
                corrected = expanded + letter + rest
    
    # --- Fix common single-character confusions ---
    # These happen at specific positions in the plate
    # E↔F confusion: In the letter position of a plate, "E" is often misread for "F" and vice versa
    # We can't fix this without context, but we can note it happens
    
    # O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6 confusions
    # For the LETTER position (3rd character in Vietnamese plates), prefer letters
    # For the NUMBER positions, prefer digits
    if len(corrected) >= 3:
        parts = list(corrected)
        # In Vietnamese plates, position index 2 (0-indexed) is always a letter
        letter_pos = None
        for i, c in enumerate(parts):
            if c.isalpha():
                letter_pos = i
                break
        
        if letter_pos is not None:
            # Characters before the letter position should be digits
            for i in range(letter_pos):
                if parts[i] == 'O': parts[i] = '0'
                elif parts[i] == 'I': parts[i] = '1'
                elif parts[i] == 'S': parts[i] = '5'
                elif parts[i] == 'B': parts[i] = '8'
                elif parts[i] == 'Z': parts[i] = '2'
                elif parts[i] == 'G': parts[i] = '6'
                elif parts[i] == 'T': parts[i] = '7'
                elif parts[i] == 'A': parts[i] = '4'
            
            # Characters after the letter position should be digits (with . and - allowed)
            for i in range(letter_pos + 1, len(parts)):
                if parts[i] == 'O': parts[i] = '0'
                elif parts[i] == 'I': parts[i] = '1'
                elif parts[i] == 'l': parts[i] = '1'
                elif parts[i] == 'S': parts[i] = '5'
                elif parts[i] == 'B': parts[i] = '8'
                elif parts[i] == 'Z': parts[i] = '2'
                elif parts[i] == 'G': parts[i] = '6'
                elif parts[i] == 'T': parts[i] = '7'
                elif parts[i] == 'A': parts[i] = '4'
                elif parts[i] == 'D': parts[i] = '0'
        
        corrected = ''.join(parts)
    
    return corrected

def format_plate_text(text):
    """Try to format the plate text into a standard Vietnamese plate format: ##X-###.##"""
    import re as _re
    
    # Remove all separators to get raw alphanumeric
    raw = text.replace('-', '').replace('.', '').replace(' ', '')
    
    if len(raw) < 7:
        return text  # Too short to format
    
    # Try to match Vietnamese plate pattern: 2 digits + 1 letter + 3-5 digits
    match = _re.match(r'^(\d{2})([A-Z]\d?)(\d{3,5})$', raw)
    if match:
        prefix = match.group(1)     # Province code (e.g., "51")
        series = match.group(2)     # Series letter (e.g., "F")
        numbers = match.group(3)    # Number portion
        
        # Format as ##X-###.## if number portion has 5+ digits
        if len(numbers) >= 5:
            return f"{prefix}{series}-{numbers[:3]}.{numbers[3:]}"
        elif len(numbers) >= 3:
            return f"{prefix}{series}-{numbers}"
    
    return text

def is_valid_plate(text):
    """Check if text looks like a valid license plate (supports formats with - and .)"""
    # Strip separators for length/content check
    core = text.replace('-', '').replace('.', '').replace(' ', '')
    if len(core) < 5 or len(core) > 12:
        return False
    letters = sum(1 for c in core if c.isalpha())
    numbers = sum(1 for c in core if c.isdigit())
    # A valid plate needs at least 1 letter and 2 numbers
    return letters >= 1 and numbers >= 2

def sort_ocr_results_reading_order(ocr_results):
    """Sort OCR results in reading order: top-to-bottom first, then left-to-right.
    This handles two-line plates (common in Asian countries)."""
    if not ocr_results:
        return ocr_results
    
    # Get vertical midpoints of each result
    midpoints = []
    for (bbox, text, prob) in ocr_results:
        y_mid = (bbox[0][1] + bbox[2][1]) / 2
        x_mid = (bbox[0][0] + bbox[2][0]) / 2
        midpoints.append((y_mid, x_mid))
    
    if len(midpoints) <= 1:
        return ocr_results
    
    # Calculate the height of the tallest bounding box to detect line breaks
    heights = [(bbox[2][1] - bbox[0][1]) for (bbox, _, _) in ocr_results]
    avg_height = sum(heights) / len(heights) if heights else 20
    
    # Group into lines: if two results are within half the avg char height, they're on the same line
    line_threshold = avg_height * 0.6
    
    # Create list of (y_mid, x_mid, index)
    indexed = [(midpoints[i][0], midpoints[i][1], i) for i in range(len(ocr_results))]
    # Sort by y first, then by x
    indexed.sort(key=lambda item: (item[0] // line_threshold, item[1]))
    
    return [ocr_results[idx] for (_, _, idx) in indexed]

def run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST):
    """Run EasyOCR on a preprocessed image with optimized parameters."""
    try:
        ocr_results = ocr_reader.readtext(
            img,
            allowlist=allowlist,
            paragraph=False,
            min_size=5,
            text_threshold=0.4,
            low_text=0.3,
            width_ths=0.3,       # CRITICAL: prevent merging adjacent character boxes
            mag_ratio=1.0,       # Fast mode without internal upscale
            slope_ths=0.2,       # Allow slight rotation
        )
        return ocr_results if ocr_results else []
    except Exception:
        return []

def extract_text_from_ocr_results(ocr_results):
    """Extract and concatenate text from OCR results in reading order."""
    if not ocr_results:
        return "", 0
    
    ocr_results = sort_ocr_results_reading_order(ocr_results)
    
    parts = []
    total_conf = 0
    count = 0
    
    for (bbox, text, prob) in ocr_results:
        if prob > 0.15:
            cleaned = clean_plate_text(text)
            if cleaned:
                parts.append(cleaned)
                total_conf += prob
                count += 1
    
    if not parts:
        return "", 0
    
    avg_conf = total_conf / count if count > 0 else 0
    
    # Determine if this is a two-line plate by checking y-coordinates
    if len(ocr_results) >= 2:
        y_positions = [(bbox[0][1] + bbox[2][1]) / 2 for (bbox, _, _) in ocr_results]
        heights = [(bbox[2][1] - bbox[0][1]) for (bbox, _, _) in ocr_results]
        avg_h = sum(heights) / len(heights) if heights else 1
        
        # If there's a significant vertical gap between results, it's two lines
        if max(y_positions) - min(y_positions) > avg_h * 0.5 and len(parts) == 2:
            concatenated = parts[0] + '-' + parts[1]
        else:
            concatenated = ''.join(parts)
    else:
        concatenated = ''.join(parts)
    
    return concatenated, avg_conf

def ocr_plate_image(plate_img_color):
    """Run OCR on a plate image using multiple preprocessing strategies, 
    character separation, and post-processing corrections."""
    if plate_img_color is None or plate_img_color.size == 0:
        return "UNKNOWN"
    
    h, w = plate_img_color.shape[:2]
    if h < 5 or w < 10:
        return "UNKNOWN"
    
    # Tier 1: Try optimal scale and 2 best pre-processing methods
    scale = 300 / w if w > 0 else 2.0
    if scale < 0.8:
        scale = 1.0  # Avoid heavy downscaling
    
    scaled = cv2.resize(plate_img_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    
    preprocessed_tier1 = []
    
    # 1. CLAHE with bilateral filter
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bfilter)
    preprocessed_tier1.append(enhanced)
    
    # 2. Otsu binarization
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_tier1.append(otsu)
    
    all_candidates = []
    
    # Run Tier 1
    for img in preprocessed_tier1:
        results = run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST)
        text, conf = extract_text_from_ocr_results(results)
        if text and len(text) >= 4:
            all_candidates.append((text, conf))
            
    # Early exit if we found a high confidence valid plate
    for text, conf in all_candidates:
        if conf > 0.8 and is_valid_plate(text):
            corrected = apply_ocr_corrections(text)
            formatted = format_plate_text(corrected)
            # Final cleanup
            best = formatted
            while '--' in best: best = best.replace('--', '-')
            while '..' in best: best = best.replace('..', '.')
            return best.strip('-.')
    
    # Tier 2: If Tier 1 failed or had low confidence, try more methods
    preprocessed_tier2 = []
    
    # 3. Morphological opening to SEPARATE touching characters (key fix!)
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_sep)
    preprocessed_tier2.append(morph_opened)
    
    # 4. Sharpened CLAHE
    sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen)
    preprocessed_tier2.append(sharpened)
    
    for img in preprocessed_tier2:
        results = run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST)
        text, conf = extract_text_from_ocr_results(results)
        if text and len(text) >= 4:
            all_candidates.append((text, conf))
            
    # Also try strict mode (no separators) on Otsu
    results_strict = run_ocr_on_image(otsu, allowlist=PLATE_ALLOWLIST_STRICT)
    text_strict, conf_strict = extract_text_from_ocr_results(results_strict)
    if text_strict and len(text_strict) >= 4:
        all_candidates.append((text_strict, conf_strict))
    
    if not all_candidates:
        return "UNKNOWN"
    
    # --- Apply corrections to all candidates ---
    corrected_candidates = []
    for text, conf in all_candidates:
        corrected = apply_ocr_corrections(text)
        formatted = format_plate_text(corrected)
        corrected_candidates.append((formatted, conf))
        # Also keep the unformatted corrected version
        if corrected != formatted:
            corrected_candidates.append((corrected, conf * 0.9))
            
    # --- Score and pick the best candidate ---
    import re as _re
    scored = []
    for text, conf in corrected_candidates:
        score = conf
        raw = text.replace('-', '').replace('.', '')
        
        # Bonus for matching Vietnamese plate format
        if _re.match(r'^\d{2}[A-Z]\d?-\d{3}\.\d{2}$', text):
            score += 2.0  
        elif _re.match(r'^\d{2}[A-Z]', raw):
            score += 0.5  
        
        if is_valid_plate(text):
            score += 0.5
        
        if 7 <= len(text) <= 12:
            score += 0.3
        
        if text.startswith('0') or text.startswith('00'):
            score -= 0.5
        
        scored.append((text, score, conf))
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    
    # Final cleanup
    while '--' in best: best = best.replace('--', '-')
    while '..' in best: best = best.replace('..', '.')
    best = best.strip('-.')
    
    return best if len(best) >= 4 else "UNKNOWN"

def extract_plate_text(vehicle_crop):
    """Attempt to find a license plate in the vehicle crop and OCR it."""
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    
    plate_img = vehicle_crop
    plate_found = False
    
    # 1. Try Custom YOLO Model First (pick highest confidence detection)
    if plate_detector is not None:
        results = plate_detector(vehicle_crop, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            # Pick the detection with the highest confidence
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())
            
            # Add padding around the plate for better OCR (5% each side)
            h_crop, w_crop = vehicle_crop.shape[:2]
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.08)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_crop, x2 + pad_x)
            y2 = min(h_crop, y2 + pad_y)
            
            plate_img = vehicle_crop[y1:y2, x1:x2]
            plate_found = True
            
    # 2. Try Haar Cascade if YOLO wasn't loaded or found nothing
    if not plate_found:
        plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
        if len(plates) > 0:
            plates = sorted(plates, key=lambda x: x[2]*x[3], reverse=True)
            px, py, pw, ph = plates[0]
            # More generous padding for Haar cascade
            px = max(0, px - 10)
            py = max(0, py - 10)
            pw = min(vehicle_crop.shape[1] - px, pw + 20)
            ph = min(vehicle_crop.shape[0] - py, ph + 20)
            plate_img = vehicle_crop[py:py+ph, px:px+pw]
            plate_found = True
            
    # 3. Fallback: crop the bottom half, avoiding extreme left/right edges
    if not plate_found:
        h, w = vehicle_crop.shape[:2]
        plate_img = vehicle_crop[int(h*0.5):h, int(w*0.1):int(w*0.9)]
        
    if plate_img.size == 0:
        return "UNKNOWN"

    return ocr_plate_image(plate_img)

# Directories
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

DETECTED_VEHICLES_FOLDER = os.path.join(BASE_PATH, "detected_vehicles")
os.makedirs(DETECTED_VEHICLES_FOLDER, exist_ok=True)

# Load known faces embeddings & pre-normalize
known_faces = {}
normalized_known_faces = {}

def reload_known_faces():
    global known_faces, normalized_known_faces
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
            new_norm_faces = {}
            for person, embeddings in known_faces.items():
                if embeddings and len(embeddings) > 0:
                    embs = np.array(embeddings)
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    new_norm_faces[person] = embs / norms
            normalized_known_faces = new_norm_faces
            print(f"[FaceRecognition] Loaded {len(normalized_known_faces)} registered suspects from {EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"[FaceRecognition] Error reloading embeddings: {e}")

reload_known_faces()

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(img, detector=None):
    if img is None or img.size == 0:
        return []
    if detector is None:
        detector = face_detector

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = []

    # Primary: MediaPipe Face Detector
    try:
        results = detector.process(img_rgb)
        if results and results.detections:
            h, w, _ = img.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, width, height = (
                    int(bboxC.xmin * w),
                    int(bboxC.ymin * h),
                    int(bboxC.width * w),
                    int(bboxC.height * h),
                )
                x, y = max(0, x), max(0, y)
                width, height = min(w - x, width), min(h - y, height)
                if width > 15 and height > 15:
                    face = img_rgb[y : y + height, x : x + width]
                    faces.append((face, (x, y, width, height)))
    except Exception:
        pass

    # Fallback: Haar Cascade if primary found no faces
    if not faces:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            h, w = img.shape[:2]
            for (fx, fy, fw, fh) in detected:
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(w - fx, fw), min(h - fy, fh)
                if fw > 15 and fh > 15:
                    face = img_rgb[fy : fy + fh, fx : fx + fw]
                    faces.append((face, (fx, fy, fw, fh)))
        except Exception:
            pass

    return faces

def recognize_face(face_embedding, threshold=0.92):
    if face_embedding is None or len(face_embedding) == 0:
        return "Unknown", 999.0
    norm = np.linalg.norm(face_embedding)
    if norm == 0:
        return "Unknown", 999.0
    face_embedding = face_embedding / norm
    
    min_dist = float("inf")
    name = "Unknown"

    for person, norm_embs in normalized_known_faces.items():
        if len(norm_embs) == 0:
            continue
        diffs = norm_embs - face_embedding
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < min_dist:
            min_dist = dists[min_idx]
            if dists[min_idx] < threshold:
                name = person

    return name, min_dist

alerts = []
last_detection_time = {}

vehicle_alerts = []
last_vehicle_detection_time = {}

def extract_faces_from_video(name, video_path, num_images=500):
    if not os.path.exists(video_path):
        return "Error: Video file does not exist."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_images)

    output_dir = os.path.join(DATASET_FOLDER, name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_images = 0
    new_embeddings = []

    local_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        frame_count += 1
        faces = extract_face(frame, detector=local_detector)

        for face, _ in faces:
            face_resized = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face_resized])[0]
            new_embeddings.append(embedding)

            image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            saved_images += 1

    cap.release()

    # Save embeddings to embeddings.pkl
    if new_embeddings:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        if name in known_faces:
            known_faces[name].extend(new_embeddings)
        else:
            known_faces[name] = new_embeddings

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        reload_known_faces()

        return f"Extracted {len(new_embeddings)} face embeddings for {name} and saved in {EMBEDDINGS_FILE}."
    
    return "No faces detected in the video."

def make_alert_call(name, timestamp):
    """Make a phone call alert when a suspect is detected"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create a TwiML response with text-to-speech
        twiml = f"""
        <Response>
            <Say>Alert! Suspect {name} has been detected at {timestamp}. Please check your email for more details.</Say>
            <Pause length="1"/>
            <Say>Repeating: Suspect {name} has been detected.</Say>
        </Response>
        """
        
        # Make the call
        call = client.calls.create(
            twiml=twiml,
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"Phone alert initiated for suspect: {name}, Call SID: {call.sid}")
        return True
    except Exception as e:
        print(f"Error making phone call: {e}")
        return False

def send_email_alert(name, timestamp, face_path):
    subject = f"Suspect Detected  : {name}"
    body = f"A suspect has been detected!!!!.\n\nName: {name}\nTime: {timestamp}"

    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach the detected face image
    try:
        with open(face_path, 'rb') as img:
            img_data = img.read()
            img_name = os.path.basename(face_path)
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)
    except Exception as e:
        print(f"Error attaching image: {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"Email alert sent for suspect: {name}")
    except Exception as e:
        print(f"Email sending failed: {e}")

# Update in generate_frames() to send email alerts
def process_alerts():
    """Thread to process suspect alerts asynchronously"""
    while True:
        task = task_queue.get()
        if task is None:
            break  # Stop the thread when None is added to the queue

        name, timestamp, face_path = task
        try:
            # Save suspect in MongoDB
            with open(face_path, "rb") as img_file:
                image_data = img_file.read()

            suspect_data = {
                "suspect_name": name,
                "detected_image": image_data,
                "time": timestamp
            }
            collection.insert_one(suspect_data)

            # Send Email
            send_email_alert(name, timestamp, face_path)
            
            # Make phone call alert
            make_alert_call(name, timestamp)

        except Exception as e:
            print(f"Error processing alert: {e}")

        task_queue.task_done()

# Start the background thread
alert_thread = threading.Thread(target=process_alerts, daemon=True)
alert_thread.start()

def process_vehicles():
    """Thread to process vehicle tracking asynchronously"""
    while True:
        task = vehicle_queue.get()
        if task is None:
            break
        
        number_plate, timestamp, vehicle_img_path = task
        try:
            with open(vehicle_img_path, "rb") as img_file:
                image_data = img_file.read()

            vehicle_data = {
                "number_plate": number_plate,
                "detected_image": image_data,
                "time": timestamp
            }
            vehicles_collection.insert_one(vehicle_data)
            print(f"Saved vehicle {number_plate} to DB.")
        except Exception as e:
            print(f"Error processing vehicle: {e}")
            
        vehicle_queue.task_done()

vehicle_thread = threading.Thread(target=process_vehicles, daemon=True)
vehicle_thread.start()

# Decoupled Multi-Camera JPEG Buffer Storage
latest_jpeg_buffers = {}
camera_worker_threads = {}
active_camera_faces = {}

def camera_stream_worker(camera_id):
    """
    Dedicated independent background worker thread per camera channel.
    Continuously reads frames for camera_id, runs detection/tracking,
    renders annotations, and updates in-memory JPEG frame buffers.
    """
    import numpy as np
    global current_tracking_stats, known_faces, last_detection_time, alerts
    frame_counter = 0

    print(f"[CameraWorker] Started dedicated worker thread for {camera_id}")

    while True:
        try:
            cam_source = camera_manager.sources.get(camera_id)
            if not cam_source or not cam_source.enabled:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(placeholder, (20, 20), (620, 460), (40, 40, 40), 2)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(placeholder, f"SENTINEL-AI STREAM: {camera_id}", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(placeholder, "STATUS: CAMERA STOPPED / OFFLINE", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(placeholder, f"TIMESTAMP: {timestamp_str}", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    latest_jpeg_buffers[camera_id] = buffer.tobytes()
                time.sleep(0.1)
                continue

            success, frame = camera_manager.read_frame(camera_id)
            if not success or frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(placeholder, (20, 20), (620, 460), (40, 40, 40), 2)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(placeholder, f"SENTINEL-AI STREAM: {camera_id}", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(placeholder, "STATUS: CAMERA OFFLINE / STANDBY", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                cv2.putText(placeholder, f"TIMESTAMP: {timestamp_str}", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    latest_jpeg_buffers[camera_id] = buffer.tobytes()
                time.sleep(0.04)
                continue

            if camera_id == "CAM01":
                frame = cv2.flip(frame, 1)

            frame_counter += 1

            # Face Recognition across ALL camera streams (throttled to every 3 frames)
            if frame_counter % 3 == 0:
                try:
                    faces = extract_face(frame)
                    current_cam_faces = []
                    for face, (x, y, width, height) in faces:
                        face_resized = cv2.resize(face, (160, 160))
                        face_embedding = embedder.embeddings([face_resized])[0]
                        name, dist = recognize_face(face_embedding)
                        current_cam_faces.append({
                            "bbox": (x, y, width, height),
                            "name": name,
                            "dist": dist,
                            "face_crop": face
                        })

                        if name != "Unknown":
                            current_time = time.time()
                            if name not in last_detection_time or (current_time - last_detection_time[name] >= 5):
                                last_detection_time[name] = current_time
                                timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                                face_filename = f"{name}_{timestamp.replace(' ', '_').replace('-', '')}.jpg"
                                face_path = os.path.join(DETECTED_FACES_FOLDER, face_filename)
                                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                                time_data_path = os.path.join(TIME_DATA_FOLDER, f"{name}.txt")
                                with open(time_data_path, "a") as f:
                                    f.write(f"{timestamp}\n")

                                alerts.append({"name": name, "time": timestamp, "camera": camera_id})
                                task_queue.put((name, timestamp, face_path))
                                print(f"[SUSPECT ALERT] {camera_id}: Suspect '{name}' DETECTED! (Distance={dist:.2f})")

                    active_camera_faces[camera_id] = current_cam_faces
                except Exception as face_err:
                    pass

            # Build spatial suspect_map mapping local track IDs to recognized suspect names
            track_suspect_map = {}
            cached_faces = active_camera_faces.get(camera_id, [])
            tracker_inst = human_tracker_engine.trackers.get(camera_id)
            if tracker_inst and cached_faces:
                active_tracks = getattr(tracker_inst, 'tracked_stracks', getattr(tracker_inst, 'tracked_tracks', []))
                for track in active_tracks:
                    t_bbox = getattr(track, 'tlbr', getattr(track, 'bbox', None))
                    if t_bbox is not None:
                        tx1, ty1, tx2, ty2 = map(int, t_bbox)
                        for face_info in cached_faces:
                            if face_info["name"] != "Unknown":
                                fx, fy, fw, fh = face_info["bbox"]
                                if fx >= tx1 - 30 and fx <= tx2 + 30 and fy >= ty1 - 30 and fy <= ty2 + 30:
                                    track_suspect_map[track.track_id] = face_info["name"]

            # Run High-Accuracy Human Detection, ByteTrack Multi-Object Tracking & Trajectory Renderer
            try:
                src_type = "live" if camera_id == "CAM01" else "file"
                human_tracks, non_human_objects, stats = human_tracker_engine.process_frame(
                    frame, camera_id=camera_id, source_type=src_type, suspect_map=track_suspect_map
                )
                current_tracking_stats = stats
                frame = surveillance_renderer.draw_annotations(frame, human_tracks, non_human_objects, stats)
            except Exception as trk_err:
                print(f"[Tracker/Renderer Error] {camera_id}: {trk_err}")

            # Draw face detection & suspect overlays on frame
            cached_faces = active_camera_faces.get(camera_id, [])
            for face_info in cached_faces:
                x, y, width, height = face_info["bbox"]
                name = face_info["name"]
                dist = face_info["dist"]

                if name != "Unknown":
                    # Red glowing suspect box & banner
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)
                    banner_text = f"SUSPECT: {name.upper()}"
                    (w_t, h_t), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x, max(0, y - 25)), (x + w_t + 10, y), (0, 0, 200), -1)
                    cv2.putText(frame, banner_text, (x + 5, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                else:
                    # Cyan/Green face box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    cv2.putText(frame, "FACE", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Safely encode annotated frame to JPEG buffer
            if frame is not None and frame.size > 0:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    latest_jpeg_buffers[camera_id] = buffer.tobytes()

        except Exception as cam_err:
            print(f"[CameraWorker] {camera_id} Exception: {cam_err}")
        time.sleep(0.02)

def start_camera_workers():
    """Ensure dedicated background worker threads are running for all configured cameras."""
    for cid in ["CAM01", "CAM02", "CAM03", "CAM04"]:
        if cid not in camera_worker_threads or not camera_worker_threads[cid].is_alive():
            t = threading.Thread(target=camera_stream_worker, args=(cid,), daemon=True)
            t.start()
            camera_worker_threads[cid] = t

# Start dedicated per-camera background worker threads
start_camera_workers()

def generate_frames(camera_id="CAM01"):
    """Lightweight 0-latency HTTP MJPEG stream generator."""
    camera_manager.start_camera(camera_id)
    while True:
        cam_source = camera_manager.sources.get(camera_id)
        if cam_source and not cam_source.enabled:
            break

        jpeg_bytes = latest_jpeg_buffers.get(camera_id)
        if jpeg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        else:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(placeholder, (20, 20), (620, 460), (40, 40, 40), 2)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(placeholder, f"SENTINEL-AI STREAM: {camera_id}", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(placeholder, "STATUS: INITIALIZING STREAM...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.putText(placeholder, f"TIMESTAMP: {timestamp_str}", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/api/tracking_stats')
def api_tracking_stats():
    global current_tracking_stats
    return jsonify(current_tracking_stats)

@app.route('/api/global_person_analytics')
def api_global_person_analytics():
    analytics = human_tracker_engine.global_id_manager.get_summary_analytics()
    return jsonify(analytics)

@app.route('/api/merge_identities', methods=['POST'])
def api_merge_identities():
    merged_count = human_tracker_engine.global_id_manager.run_merge_pass(merge_threshold=0.75)
    analytics = human_tracker_engine.global_id_manager.get_summary_analytics()
    return jsonify({"success": True, "merged_count": merged_count, "analytics": analytics})

@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)

@app.route('/suspects')
def get_suspects():
    # Find all suspects in the database
    suspects = list(collection.find({}))
    # Convert the ObjectId to string for each document
    for suspect in suspects:
        suspect['_id'] = str(suspect['_id'])
        # Convert binary image data to base64 for display in HTML
        if 'detected_image' in suspect:
            suspect['image_b64'] = base64.b64encode(suspect['detected_image']).decode('utf-8')
    
    return render_template('suspects.html', suspects=suspects)

@app.route('/vehicles')
def get_vehicles():
    # Find all vehicles in the database
    vehicles = list(vehicles_collection.find({}))
    for vehicle in vehicles:
        if '_id' in vehicle:
            vehicle['_id'] = str(vehicle['_id'])
        if 'detected_image' in vehicle:
            vehicle['image_b64'] = base64.b64encode(vehicle['detected_image']).decode('utf-8')
    
    return render_template('vehicles.html', vehicles=vehicles)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/live-mon')
@app.route('/live-monitoring')
def live_mon():
    global alerts
    alerts = []  # Clear previous alerts
    return render_template("livemon.html")

@app.route('/multicam')
def multicam():
    return render_template("multicam.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames("CAM01"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<camera_id>')
def video_feed_cam(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
def api_cameras():
    return jsonify(camera_manager.get_active_cameras())

@app.route('/api/cameras/<camera_id>/start', methods=['POST'])
def api_start_camera(camera_id):
    success = camera_manager.start_camera(camera_id)
    return jsonify({"camera_id": camera_id, "success": success})

@app.route('/api/cameras/<camera_id>/stop', methods=['POST'])
def api_stop_camera(camera_id):
    success = camera_manager.stop_camera(camera_id)
    return jsonify({"camera_id": camera_id, "success": success})

@app.route('/api/cameras/<camera_id>/pause', methods=['POST'])
def api_pause_camera(camera_id):
    success = camera_manager.pause_camera(camera_id)
    return jsonify({"camera_id": camera_id, "success": success})

@app.route('/api/cameras/<camera_id>/resume', methods=['POST'])
def api_resume_camera(camera_id):
    success = camera_manager.resume_camera(camera_id)
    return jsonify({"camera_id": camera_id, "success": success})

@app.route('/api/cameras/<camera_id>/restart', methods=['POST'])
def api_restart_camera(camera_id):
    success = camera_manager.restart_camera(camera_id)
    return jsonify({"camera_id": camera_id, "success": success})

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/new-crim', methods=['GET', 'POST'])
def newcrim():
    if request.method == 'POST':
        name = request.form.get("name")
        file = request.files["video"]

        if file and name:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = extract_faces_from_video(name, filepath)
            flash(result)
            return redirect(url_for('newcrim'))

    return render_template('newcrim.html')

def analyze_video_file(video_path):
    """Analyze a video file for registered faces, unknown faces, and vehicle plates."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Process every Nth frame to keep analysis fast
    step = max(1, total_frames // 30)  # analyze ~30 frames max

    registered_faces = []
    unknown_faces = []
    vehicles = []

    seen_names = set()       # avoid duplicates for registered faces
    seen_unknown = 0         # cap unknown faces shown
    seen_plates = set()      # avoid duplicate plates

    local_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Reload known faces
    current_known = {}
    normalized_current = {}
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            current_known = pickle.load(f)
            
        for person, embeddings in current_known.items():
            if embeddings:
                embs = np.array(embeddings)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_current[person] = embs / norms

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        frame_count += 1
        
        # Resize frame to max 720p for faster processing
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # --- Face Detection ---
        faces = extract_face(frame, detector=local_detector)
        
        valid_faces = []
        face_images = []
        for face, bbox in faces:
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue
            face_resized = cv2.resize(face, (160, 160))
            valid_faces.append((face_resized, bbox))
            face_images.append(face_resized)
            
        if face_images:
            # Batch embedding
            embeddings_batch = embedder.embeddings(face_images)
            
            for i, ((face_resized, _), face_embedding) in enumerate(zip(valid_faces, embeddings_batch)):
                norm = np.linalg.norm(face_embedding)
                if norm > 0:
                    face_embedding = face_embedding / norm
                
                min_dist = float('inf')
                best_name = 'Unknown'
                
                for person, norm_embs in normalized_current.items():
                    if len(norm_embs) == 0:
                        continue
                    diffs = norm_embs - face_embedding
                    dists = np.linalg.norm(diffs, axis=1)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < 0.7 and dists[min_idx] < min_dist:
                        min_dist = dists[min_idx]
                        best_name = person

                # Encode face image to base64
                face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode('.jpg', face_bgr)
                img_b64 = base64.b64encode(buf).decode('utf-8')

                if best_name != 'Unknown':
                    if best_name not in seen_names:
                        seen_names.add(best_name)
                        registered_faces.append({
                            'name': best_name,
                            'image_b64': img_b64,
                            'frame_number': frame_count
                        })
                else:
                    if seen_unknown < 20:  # cap at 20 unknown faces
                        seen_unknown += 1
                        unknown_faces.append({
                            'name': 'Unknown',
                            'image_b64': img_b64,
                            'frame_number': frame_count
                        })

        # --- Vehicle Detection ---
        try:
            results = vehicle_model(frame, device=device, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        vehicle_crop = frame[y1:y2, x1:x2]
                        if vehicle_crop.size > 0:
                            plate_text = extract_plate_text(vehicle_crop)

                            if plate_text == "UNKNOWN":
                                plate_text = f"UNKNOWN_{frame_count}"

                            if plate_text not in seen_plates:
                                seen_plates.add(plate_text)
                                # Draw bbox on the crop for display
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, plate_text, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                # Encode the full frame with bbox as image
                                vehicle_display = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                                if vehicle_display.size > 0:
                                    _, vbuf = cv2.imencode('.jpg', vehicle_display)
                                    v_b64 = base64.b64encode(vbuf).decode('utf-8')
                                    vehicles.append({
                                        'plate': plate_text,
                                        'image_b64': v_b64,
                                        'frame_number': frame_count
                                    })
        except Exception as e:
            print(f"Vehicle detection error in frame {frame_count}: {e}")

    cap.release()
    local_detector.close()

    return {
        'registered_faces': registered_faces,
        'unknown_faces': unknown_faces,
        'vehicles': vehicles
    }


@app.route('/analyze-video', methods=['GET', 'POST'])
def analyze_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, 'analyze_' + file.filename)
            file.save(filepath)

            results = analyze_video_file(filepath)
            if results is None:
                flash('Error: Could not open the video file.')
                return redirect(url_for('analyze_video'))

            # Clean up uploaded file after processing
            try:
                os.remove(filepath)
            except:
                pass

            return render_template('analyze_video.html', results=results)
        else:
            flash('Please select a video file.')
            return redirect(url_for('analyze_video'))

    return render_template('analyze_video.html', results=None)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5000)