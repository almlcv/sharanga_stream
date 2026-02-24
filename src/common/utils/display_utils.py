# import cv2
# import numpy as np

# def draw_detections(frame, results, model_id):
#     """Draw YOLO detection boxes and class names only"""
    
#     # Color per model
#     colors = [
#         (0, 255, 0),    # Model 0 (fire_smoke)
#         (255, 165, 0),  # Model 1 (ppe)
#         (255, 0, 255)   # Model 2 (load_unload)
#     ]
#     color = colors[model_id % len(colors)]

#     if results and len(results) > 0:
#         boxes = results[0].boxes
        
#         for box in boxes:
#             try:
#                 # Get bbox coordinates
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 cls = int(box.cls[0])
#                 label = results[0].names[cls]
                
#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
#                 # Draw class name label
#                 (label_width, label_height), _ = cv2.getTextSize(
#                     label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
#                 )
#                 cv2.rectangle(frame, (x1, y1 - label_height - 10), 
#                              (x1 + label_width + 10, y1), color, -1)
#                 cv2.putText(frame, label, (x1 + 5, y1 - 5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             except Exception:
#                 continue
    
#     return frame