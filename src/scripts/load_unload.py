import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from threading import Lock
from src.common.utils.db import db

# MongoDB Configuration
COLLECTION_NAME = "LoadingUnloading"
collection = db[COLLECTION_NAME]

# Directory to save detected frames
SAVE_DIR = "detected_frames/load_unload"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# ROI CONFIGURATION - One vehicle per ROI at a time
# ============================================================================
STREAM_ROIS = {
    "UnloadZone": [(67, 359), (164, 236), (256, 138), (503, 214), (502, 238), (542, 259), (514, 590), (449, 563), (436, 603), (421, 640), (346, 605), (247, 542), (179, 480), (116, 416)],
    "LoadZone": [(113, 348), (98, 176), (97, 86), (96, 1), (344, 1), (341, 152), (345, 272)] 
}

# ============================================================================
# THRESHOLDS FOR SINGLE VEHICLE TRACKING  (all time-based, in seconds)
# ============================================================================

# Minimum *duration* (seconds) the vehicle must be present to save to DB.
# Prevents saving spurious short detections.
MINIMUM_DURATION_SECONDS = 30

# How many seconds with ZERO detections in the ROI before we decide the
# vehicle has truly exited.  Set this LONGER than the worst-case occlusion.
# e.g. 5 minutes = 300 s  →  a 3-4 min occlusion will NOT trigger a false exit.
COMPLETE_EXIT_SECONDS = 30

# ============================================================================
# THREAD-SAFE DATA STRUCTURES
# ============================================================================
VEHICLE_TRACKS_LOCK = Lock()
FRAME_COUNTER_LOCK = Lock()
MONGODB_LOCK = Lock()

# Single vehicle session per stream – we do NOT care about track IDs.
CURRENT_VEHICLE: Dict[str, Optional[dict]] = {}
STREAM_FRAME_COUNTERS: Dict[str, int] = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stream_roi(stream_name: Optional[str], stream_index: int) -> Optional[List[Tuple[int, int]]]:
    """Get ROI for a stream"""
    if stream_name and stream_name in STREAM_ROIS:
        return STREAM_ROIS[stream_name]
    
    # Return first ROI as fallback
    if STREAM_ROIS:
        return list(STREAM_ROIS.values())[stream_index % len(STREAM_ROIS)]
    
    return None


def get_stream_key(stream_name: Optional[str], stream_index: int) -> str:
    """Generate stream key"""
    return stream_name if stream_name else f"stream_{stream_index}"


def save_to_mongodb(data: dict) -> bool:
    """Thread-safe MongoDB save"""
    if collection is None:
        print("[!] MongoDB not available")
        return False
    
    with MONGODB_LOCK:
        try:
            result = collection.insert_one({
                "stream_name": data["stream_name"],
                "entry_time": data["entry_time"],
                "exit_time": data["exit_time"],
                "duration_seconds": data["duration_seconds"],
                "entry_image_path": data["entry_image_path"],
                "exit_image_path": data["exit_image_path"],
                "timestamp": datetime.now()
            })
            print(f"[✓] MongoDB ID: {result.inserted_id}")
            return True
        except Exception as e:
            print(f"[!] MongoDB error: {e}")
            return False


def save_image(frame, stream_key: str, event_type: str) -> str:
    """Save image and return absolute path"""
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    save_dir = os.path.join(SAVE_DIR, stream_key, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    # only timestamp in filename
    timestamp_file = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp_file}.jpg"
    relative_path = os.path.join(save_dir, filename)

    # convert to absolute path
    full_path = os.path.abspath(relative_path)

    cv2.imwrite(full_path, frame)
    return full_path


def draw_roi(frame, roi, border_color=(0, 215, 255), thickness=2):
    """Draw ROI polygon"""
    if not roi:
        return
    pts = np.array(roi, np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=border_color, 
                  thickness=thickness, lineType=cv2.LINE_AA)


def point_in_roi(point: Tuple[float, float], roi: List[Tuple[int, int]]) -> bool:
    """Check if point is inside ROI"""
    if not roi:
        return True
    
    try:
        contour = np.array(roi, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0
    except Exception as e:
        print(f"[WARN] ROI test failed: {e}")
        return False


# ============================================================================
# FRAME COUNTER
# ============================================================================

def get_and_increment_frame_counter(stream_key: str) -> int:
    """Thread-safe frame counter"""
    with FRAME_COUNTER_LOCK:
        if stream_key not in STREAM_FRAME_COUNTERS:
            STREAM_FRAME_COUNTERS[stream_key] = 0
        STREAM_FRAME_COUNTERS[stream_key] += 1
        return STREAM_FRAME_COUNTERS[stream_key]


# ============================================================================
# VEHICLE SESSION MANAGEMENT
# ============================================================================

def initialize_stream_tracking(stream_key: str):
    """Initialize stream tracking"""
    with VEHICLE_TRACKS_LOCK:
        if stream_key not in CURRENT_VEHICLE:
            CURRENT_VEHICLE[stream_key] = None


def get_current_vehicle(stream_key: str) -> Optional[dict]:
    """Get current tracked vehicle session (only one per stream)"""
    with VEHICLE_TRACKS_LOCK:
        return CURRENT_VEHICLE.get(stream_key)


def set_current_vehicle(stream_key: str, vehicle_data: Optional[dict]):
    """Set current vehicle session (replaces previous)"""
    with VEHICLE_TRACKS_LOCK:
        CURRENT_VEHICLE[stream_key] = vehicle_data


def update_vehicle_seen(stream_key: str, now: datetime, track_id: int = 0):
    """Mark that the vehicle was seen this frame (regardless of track ID)"""
    with VEHICLE_TRACKS_LOCK:
        vehicle = CURRENT_VEHICLE.get(stream_key)
        if vehicle is not None:
            vehicle["last_seen_time"] = now
            vehicle["total_frames_in_roi"] += 1
            # Update display ID if we started with 0 and now have a real one
            if vehicle.get("initial_track_id", 0) == 0 and track_id > 0:
                vehicle["initial_track_id"] = track_id


def get_seconds_since_last_seen(stream_key: str, now: datetime) -> float:
    """Return seconds elapsed since the vehicle was last detected. 0 if no session."""
    with VEHICLE_TRACKS_LOCK:
        vehicle = CURRENT_VEHICLE.get(stream_key)
        if vehicle is not None:
            return (now - vehicle["last_seen_time"]).total_seconds()
    return 0.0


# ============================================================================
# VEHICLE ENTRY/EXIT
# ============================================================================

def start_vehicle_session(frame, stream_key: str, track_id: int,
                          now: datetime, current_frame: int):
    """
    Start a new vehicle session.
    We record the initial track_id for logging only; it will NOT be used
    for matching on subsequent frames.
    """
    entry_image_path = save_image(frame, stream_key, "entry")

    vehicle_data = {
        "stream_name": stream_key,
        "initial_track_id": track_id,        # for logging only
        "entry_time": now,
        "entry_frame": current_frame,
        "entry_image_path": entry_image_path,
        "last_seen_time": now,
        "total_frames_in_roi": 1,
    }

    set_current_vehicle(stream_key, vehicle_data)

    print(f"\n[ENTRY] {stream_key} - Track {track_id} | Frame {current_frame}")
    print(f"  Time: {now.strftime('%H:%M:%S.%f')[:-3]}\n")


def finalize_vehicle_exit(frame, stream_key: str, vehicle_data: dict,
                          current_frame: int, now: datetime):
    """Finalize vehicle exit and save to DB"""

    initial_id = vehicle_data.get("initial_track_id", "?")
    entry_time = vehicle_data["entry_time"]
    total_frames = vehicle_data["total_frames_in_roi"]
    duration = (now - entry_time).total_seconds()

    exit_image_path = save_image(frame, stream_key, "exit")

    print(f"\n[EXIT] {stream_key} - Initial Track {initial_id} | Frame {current_frame}")
    print(f"  Entry:    {entry_time.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  Exit:     {now.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  Duration: {duration:.2f}s | Frames: {total_frames}")

    # Save to database if minimum duration met
    if duration >= MINIMUM_DURATION_SECONDS:
        mongodb_data = {
            "stream_name": stream_key,
            "entry_time": entry_time,
            "exit_time": now,
            "duration_seconds": round(duration, 2),
            "entry_image_path": vehicle_data["entry_image_path"],
            "exit_image_path": exit_image_path
        }

        save_to_mongodb(mongodb_data)
        print(f"  [✓] SAVED TO DATABASE\n")
    else:
        print(f"  [SKIPPED] Only {duration:.1f}s (min: {MINIMUM_DURATION_SECONDS}s)\n")

    # Clear current vehicle session
    set_current_vehicle(stream_key, None)


# ============================================================================
# MAIN DETECTION PROCESSING
# ============================================================================

def process_detection(frame, results, model_id: int, stream_index: int,
                      stream_name: Optional[str] = None):
    """
    Process detection — SINGLE VEHICLE per ROI, ID-agnostic.

    Core logic:
      • If NO vehicle session is active and we detect a vehicle in the ROI,
        start a new session.
      • If a vehicle session IS active and we detect ANY vehicle in the ROI,
        treat it as the SAME vehicle (update last_seen_time).
      • If a vehicle session IS active but NOTHING is detected in the ROI,
        check elapsed time since last seen.  After COMPLETE_EXIT_SECONDS
        of no detections, finalize the exit.

    This completely eliminates tracker ID churn because we never compare IDs
    once a session is running.  Time-based exit detection tolerates long
    occlusions (3-4+ minutes) without splitting the session.
    """

    now = datetime.now()
    stream_key = get_stream_key(stream_name, stream_index)
    current_frame = get_and_increment_frame_counter(stream_key)

    # Initialize
    initialize_stream_tracking(stream_key)

    # Get ROI
    roi = get_stream_roi(stream_name, stream_index)
    if roi:
        draw_roi(frame, roi)

    # Get current tracked vehicle session
    current_vehicle = get_current_vehicle(stream_key)

    # --- Find first vehicle detection in ROI ---
    vehicle_in_roi = False
    detected_track_id: Optional[int] = None
    detected_bbox: Optional[np.ndarray] = None
    detected_center: Optional[Tuple[int, int]] = None

    if results and len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]

            if label.lower() != "vehicle":
                continue

            # We accept detections even without a track ID.
            # If the tracker assigned one, grab it for display; otherwise use 0.
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = 0

            bbox = box.xyxy[0].cpu().numpy().astype(int)
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            in_roi = (roi is None) or point_in_roi((float(center_x), float(center_y)), roi)
            if in_roi:
                vehicle_in_roi = True
                detected_track_id = track_id
                detected_bbox = bbox
                detected_center = (center_x, center_y)
                break  # Only one vehicle expected

    # --- Tracking state machine ---

    if vehicle_in_roi and detected_bbox is not None and detected_center is not None:
        x1, y1, x2, y2 = detected_bbox
        center_x, center_y = detected_center

        # Draw visualization
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if current_vehicle is None:
            # --- No active session → start one ---
            start_vehicle_session(frame, stream_key, detected_track_id or 0, now, current_frame)
        else:
            # --- Active session → same vehicle, just update ---
            update_vehicle_seen(stream_key, now, detected_track_id or 0)

        # Draw label (use session's initial ID for stable display)
        active = get_current_vehicle(stream_key)
        display_id = active["initial_track_id"] if active else (detected_track_id or 0)

        label_text = f"ID: {display_id}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1

        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        cv2.rectangle(frame, (x1, y1 - text_height - 10),
                      (x1 + text_width + 10, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5), font,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return True

    else:
        # --- No vehicle detected in ROI this frame ---
        if current_vehicle is not None:
            elapsed = get_seconds_since_last_seen(stream_key, now)

            if elapsed >= COMPLETE_EXIT_SECONDS:
                finalize_vehicle_exit(frame, stream_key, current_vehicle, current_frame, now)

        return False