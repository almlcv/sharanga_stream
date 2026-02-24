from dataclasses import dataclass
from typing import List, Optional

# --- Configuration ---
@dataclass
class RTSPStream:
    """Configuration for an RTSP stream"""
    url: str
    name: str
    channel: int
    model_id: int
      
#RTSP streams
STREAM_CONFIGS = [
    # Model 0
    RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=3&subtype=0",
        url="rtsp://127.0.0.1:8554/cam1",
        name="Shopfloor_1",
        channel=3,
        model_id=0
    ),
    RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=11&subtype=0",
        url="rtsp://127.0.0.1:8554/cam2",
        name="Store_1",
        channel=11,
        model_id=0
    ),
    RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=18&subtype=0",
        url="rtsp://127.0.0.1:8554/cam3",
        name="Shopfloor_2",
        channel=18,
        model_id=0
    ),

    RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=21&subtype=0",
        url="rtsp://127.0.0.1:8554/cam4",
        name="ToolArea",
        channel=21,
        model_id=1
    ),
    
    # Model 2
    RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=17&subtype=0",
        url="rtsp://127.0.0.1:8554/cam5",
        name="UnloadZone",
        channel=17,
        model_id=2
    ),
        RTSPStream(
        # url="rtsp://admin:rabs%40123@103.92.121.82:554/cam/realmonitor?channel=22&subtype=0",
        url="rtsp://127.0.0.1:8554/cam6",
        name="LoadZone",
        channel=22,
        model_id=2
        ),
]

# YOLO Model Configuration
@dataclass
class ModelConfig:
    model_path: str
    conf_threshold: float
    iou_threshold: float
    classes: Optional[List[int]]
    gpu_id: int
    batch_size: int


MODEL_CONFIGS = [
    ModelConfig(
        model_path="fireV1.pt", 
        conf_threshold=0.9,
        iou_threshold=0.45,
        classes=[0,2],  
        gpu_id=0,
        batch_size=4
    ),
    ModelConfig(
        model_path="ppe.pt", 
        conf_threshold=0.7,
        iou_threshold=0.5,
        classes=[0,1,2],  
        gpu_id=0,
        batch_size=4
    ),
    ModelConfig(
        model_path="load_unload.pt",  
        conf_threshold=0.25,
        iou_threshold=0.45,
        classes=[0],  
        gpu_id=0,
        batch_size=4
    ),
]

# Display configuration
STREAM_WIDTH = 640
STREAM_HEIGHT = 640

# Queue configuration
QUEUE_SIZE = 3
DETECTION_INTERVAL = 1
