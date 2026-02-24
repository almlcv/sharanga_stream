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
# OPTIMIZED THRESHOLDS FOR SINGLE VEHICLE TRACKING
# ============================================================================
MINIMUM_FRAMES_TO_SAVE = 750      
OCCLUSION_GRACE_FRAMES = 750      
COMPLETE_EXIT_FRAMES = 750        
TRACK_ID_COOLDOWN_FRAMES = 750    

# ============================================================================
# THREAD-SAFE DATA STRUCTURES
# ============================================================================
VEHICLE_TRACKS_LOCK = Lock()
TRACK_ID_HISTORY_LOCK = Lock()
FRAME_COUNTER_LOCK = Lock()
MONGODB_LOCK = Lock()

# Single vehicle tracking per stream
CURRENT_VEHICLE: Dict[str, Optional[dict]] = {}
TRACK_ID_HISTORY: Dict[str, Dict[int, int]] = {}
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
# TRACK ID COOLDOWN
# ============================================================================

def is_track_id_in_cooldown(stream_key: str, track_id: int, current_frame: int) -> bool:
    """Check if track ID is in cooldown"""
    with TRACK_ID_HISTORY_LOCK:
        if stream_key not in TRACK_ID_HISTORY:
            return False
        
        if track_id not in TRACK_ID_HISTORY[stream_key]:
            return False
        
        frames_since_exit = current_frame - TRACK_ID_HISTORY[stream_key][track_id]
        return frames_since_exit < TRACK_ID_COOLDOWN_FRAMES


def cleanup_old_track_history(stream_key: str, current_frame: int):
    """Remove expired track IDs"""
    with TRACK_ID_HISTORY_LOCK:
        if stream_key not in TRACK_ID_HISTORY:
            return
        
        expired_ids = [
            tid for tid, exit_frame in TRACK_ID_HISTORY[stream_key].items()
            if current_frame - exit_frame >= TRACK_ID_COOLDOWN_FRAMES
        ]
        
        for tid in expired_ids:
            del TRACK_ID_HISTORY[stream_key][tid]


def record_track_id_exit(stream_key: str, track_id: int, current_frame: int):
    """Record track ID exit for cooldown"""
    with TRACK_ID_HISTORY_LOCK:
        if stream_key not in TRACK_ID_HISTORY:
            TRACK_ID_HISTORY[stream_key] = {}
        TRACK_ID_HISTORY[stream_key][track_id] = current_frame


def initialize_stream_tracking(stream_key: str):
    """Initialize stream tracking"""
    with VEHICLE_TRACKS_LOCK:
        if stream_key not in CURRENT_VEHICLE:
            CURRENT_VEHICLE[stream_key] = None
    
    with TRACK_ID_HISTORY_LOCK:
        if stream_key not in TRACK_ID_HISTORY:
            TRACK_ID_HISTORY[stream_key] = {}


def get_current_vehicle(stream_key: str) -> Optional[dict]:
    """Get current tracked vehicle (only one per stream)"""
    with VEHICLE_TRACKS_LOCK:
        return CURRENT_VEHICLE.get(stream_key)


def set_current_vehicle(stream_key: str, vehicle_data: Optional[dict]):
    """Set current vehicle (replaces previous)"""
    with VEHICLE_TRACKS_LOCK:
        CURRENT_VEHICLE[stream_key] = vehicle_data


def update_vehicle_last_seen(stream_key: str, track_id: int, current_frame: int):
    """Update vehicle last seen frame and track_id - Thread-safe"""
    with VEHICLE_TRACKS_LOCK:
        if stream_key in CURRENT_VEHICLE and CURRENT_VEHICLE[stream_key] is not None:
            vehicle = CURRENT_VEHICLE[stream_key]
            if vehicle is not None:
                vehicle["track_id"] = track_id
                vehicle["last_seen_frame"] = current_frame
                vehicle["total_frames_in_roi"] += 1


# ============================================================================
# VEHICLE ENTRY/EXIT
# ============================================================================

def start_vehicle_tracking(frame, stream_key: str, track_id: int, 
                          now: datetime, current_frame: int) -> bool:
    """
    Start tracking new vehicle (only one per ROI)
    Returns True if tracking started, False if in cooldown
    """
    
    # Check cooldown
    if is_track_id_in_cooldown(stream_key, track_id, current_frame):
        print(f"[IGNORED] {stream_key} - Track {track_id} in cooldown")
        return False
    
    # Save entry image
    entry_image_path = save_image(frame, stream_key, "entry")
    
    # Create vehicle tracking data
    vehicle_data = {
        "stream_name": stream_key,
        "track_id": track_id,
        "entry_time": now,
        "entry_frame": current_frame,
        "entry_image_path": entry_image_path,
        "last_seen_frame": current_frame,
        "total_frames_in_roi": 1
    }
    
    set_current_vehicle(stream_key, vehicle_data)
    
    print(f"\n[ENTRY] {stream_key} - Track {track_id} | Frame {current_frame}")
    print(f"  Time: {now.strftime('%H:%M:%S.%f')[:-3]}\n")
    
    return True


def finalize_vehicle_exit(frame, stream_key: str, vehicle_data: dict, 
                         current_frame: int, now: datetime):
    """Finalize vehicle exit and save to DB"""
    
    track_id = vehicle_data["track_id"]
    entry_time = vehicle_data["entry_time"]
    total_frames = vehicle_data["total_frames_in_roi"]
    duration = (now - entry_time).total_seconds()
    
    # Save exit image
    exit_image_path = save_image(frame, stream_key, "exit")
    
    print(f"\n[EXIT] {stream_key} - Track {track_id} | Frame {current_frame}")
    print(f"  Entry:    {entry_time.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  Exit:     {now.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"  Duration: {duration:.2f}s | Frames: {total_frames}")
    
    # Save to database if minimum frames met
    if total_frames >= MINIMUM_FRAMES_TO_SAVE:
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
        print(f"  [SKIPPED] Only {total_frames} frames (min: {MINIMUM_FRAMES_TO_SAVE})\n")
    
    # Record exit for cooldown
    record_track_id_exit(stream_key, track_id, current_frame)
    
    # Clear current vehicle
    set_current_vehicle(stream_key, None)


# ============================================================================
# MAIN DETECTION PROCESSING
# ============================================================================

def process_detection(frame, results, model_id: int, stream_index: int, 
                     stream_name: Optional[str] = None):
    """
    Process detection - SINGLE VEHICLE per ROI
    - If vehicle reappears within 100 frames, treat as same vehicle
    - Handles ID changes due to occlusion
    - Frame drop resilient
    - Independent tracking per camera
    """
    
    now = datetime.now()
    stream_key = get_stream_key(stream_name, stream_index)
    current_frame = get_and_increment_frame_counter(stream_key)
    
    # Initialize
    initialize_stream_tracking(stream_key)
    cleanup_old_track_history(stream_key, current_frame)
    
    # Get ROI
    roi = get_stream_roi(stream_name, stream_index)
    if roi:
        draw_roi(frame, roi)
    
    # Get current tracked vehicle
    current_vehicle = get_current_vehicle(stream_key)
    
    # Handle no detections
    if not results or len(results) == 0 or len(results[0].boxes) == 0:
        handle_no_detections(frame, stream_key, current_vehicle, current_frame, now)
        return False
    
    # Process detections
    boxes = results[0].boxes
    vehicle_detected_in_roi = False
    detected_track_id: Optional[int] = None
    detected_bbox: Optional[np.ndarray] = None
    detected_center: Optional[Tuple[int, int]] = None
    
    # Find first vehicle in ROI (we expect only one)
    for box in boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        
        if label.lower() != "vehicle":
            continue
        
        if not hasattr(box, 'id') or box.id is None:
            continue
        
        track_id = int(box.id[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Check if in ROI (only if roi is not None)
        if roi is not None and point_in_roi((float(center_x), float(center_y)), roi):
            vehicle_detected_in_roi = True
            detected_track_id = track_id
            detected_bbox = bbox
            detected_center = (center_x, center_y)
            break  # Only one vehicle expected
        elif roi is None:
            # No ROI defined, accept all vehicles
            vehicle_detected_in_roi = True
            detected_track_id = track_id
            detected_bbox = bbox
            detected_center = (center_x, center_y)
            break
    
    # Process the detected vehicle
    if vehicle_detected_in_roi and detected_track_id is not None and detected_bbox is not None and detected_center is not None:
        x1, y1, x2, y2 = detected_bbox
        center_x, center_y = detected_center
        
        # Draw visualization
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Handle tracking logic
        if current_vehicle is None:
            # No vehicle being tracked - start new tracking
            tracking_started = start_vehicle_tracking(frame, stream_key, detected_track_id, now, current_frame)
            
            if not tracking_started:
                # Vehicle in cooldown, skip visualization
                return False
            
            display_id = detected_track_id
            
        elif current_vehicle["track_id"] == detected_track_id:
            # Same vehicle - update tracking (thread-safe)
            update_vehicle_last_seen(stream_key, detected_track_id, current_frame)
            display_id = detected_track_id
            
        else:
            # Different track ID detected
            frames_since_seen = current_frame - current_vehicle["last_seen_frame"]
            
            # USER FLOW: If vehicle visible within 100 frames, treat as same vehicle
            if frames_since_seen <= OCCLUSION_GRACE_FRAMES:
                # Same vehicle with new ID - update tracking (thread-safe)
                print(f"[ID CHANGE] {stream_key} - {current_vehicle['track_id']} → {detected_track_id} (within {frames_since_seen} frames)")
                
                update_vehicle_last_seen(stream_key, detected_track_id, current_frame)
                display_id = detected_track_id
                
            else:
                # Too long gap - finalize old vehicle and start new
                print(f"[NEW VEHICLE] {stream_key} - Gap of {frames_since_seen} frames")
                print(f"  Finalizing Track {current_vehicle['track_id']}, starting Track {detected_track_id}")
                
                finalize_vehicle_exit(frame, stream_key, current_vehicle, current_frame, now)
                
                tracking_started = start_vehicle_tracking(frame, stream_key, detected_track_id, now, current_frame)
                
                if not tracking_started:
                    # New vehicle in cooldown, skip visualization
                    return False
                
                display_id = detected_track_id
        
        # Draw label
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
    
    # No vehicle in ROI - check if we need to finalize exit
    else:
        if current_vehicle is not None:
            frames_since_seen = current_frame - current_vehicle["last_seen_frame"]
            
            if frames_since_seen >= COMPLETE_EXIT_FRAMES:
                finalize_vehicle_exit(frame, stream_key, current_vehicle, current_frame, now)
        
        return False


def handle_no_detections(frame, stream_key: str, current_vehicle: Optional[dict], 
                        current_frame: int, now: datetime):
    """Handle frame with no detections"""
    
    if current_vehicle is None:
        return
    
    frames_since_seen = current_frame - current_vehicle["last_seen_frame"]
    
    if frames_since_seen >= COMPLETE_EXIT_FRAMES:
        finalize_vehicle_exit(frame, stream_key, current_vehicle, current_frame, now)