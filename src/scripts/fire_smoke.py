import os
import cv2
from datetime import datetime
from typing import Optional


# Directory to save detected frames
BASE_SAVE_DIR = "detected_frames/fire_smoke"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

def process_detection(frame, results, model_id: int, stream_index: int, stream_name: Optional[str] = None):
    """
    Process detection results for fire/smoke model.
    Draw bounding boxes for all detections and save when fire/smoke detected.
    
    Args:
        frame: The original frame
        results: YOLO detection results
        model_id: ID of the model
        stream_index: Index of the stream
        stream_name: Name of the stream (optional)
    
    Returns:
        bool: True if fire/smoke was detected, False otherwise
    """
    if not results or len(results) == 0:
        return False
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return False
    
    has_fire_or_smoke = False
    
    # Draw all bounding boxes on frame (visible in stream)
    for box in boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox
        
        # Determine color based on label
        if label.lower() == "fire":
            color = (0, 0, 255)  # Red for fire
            has_fire_or_smoke = True
        elif label.lower() == "smoke":
            color = (0, 165, 255)  # Orange for smoke
            has_fire_or_smoke = True
        else:
            color = (0, 255, 0)  # Green for other detections
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_text = f"{label} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 15),
                      (x1 + label_width + 15, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 7, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save frame if fire or smoke detected
    if has_fire_or_smoke:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        save_dir = os.path.join(BASE_SAVE_DIR, date_folder)
        os.makedirs(save_dir, exist_ok=True)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"detection_stream{stream_index}_{timestamp_file}.jpg")
        cv2.imwrite(filename, frame)
    
    return has_fire_or_smoke