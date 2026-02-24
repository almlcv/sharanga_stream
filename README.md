## Features

- Real-time RTSP stream processing with GPU acceleration using FFmpeg
- Multiple model support for different detection tasks:
  - Fire and smoke detection
  - PPE (Personal Protective Equipment) detection
- Efficient multi-processing architecture for parallel stream processing
- FastAPI-based REST API
- Email alert system


## System Requirements
- FFmpeg with GPU acceleration support


## Project Structure

```
├── api/                    # FastAPI application
│   ├── core/              # Core application logic
│   ├── models/            # Data models and responses
│   └── routes/            # API endpoints
├── src/
│   ├── common/            # Core functionality
│   │   ├── gui_monitor.py     # GUI monitoring system
│   │   ├── producer.py        # Stream producer
│   │   ├── stream_capture.py  # RTSP stream capture
│   │   ├── yolo_detector.py   # YOLO detection system
│   │   └── utils/            # Utility functions
│   ├── config/           # Configuration management
│   └── scripts/          # Detection scripts
│       ├── fire_smoke.py     # Fire and smoke detection
│       └── ppe.py            # PPE detection
├── main.py              # Application entry point
└── pyproject.toml       # Project configuration and dependencies
```

## Key Components

### Stream Capture (`src/common/stream_capture.py`)
Handles RTSP stream capture using FFmpeg with GPU acceleration. Implements efficient frame reading and error handling.

### YOLO Detector (`src/common/yolo_detector.py`)
Manages object detection using YOLO models in separate processes. Supports multiple detection types:
- Fire and smoke detection
- PPE (Personal Protective Equipment) detection

### API Structure
- FastAPI-based REST API
- WebSocket support for real-time notifications
- Structured response models
- Detection route handling

## Configuration

The system uses environment variables and configuration files to manage:
- RTSP stream settings
- Model parameters (confidence threshold, IOU threshold)
- Email alert settings


## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Configure environment variables
4. Place YOLO model files (yolo11n.pt, yolo11s.pt) in the project root

## Usage

1. Start the server:
   ```bash
   fastapi dev main.py --host 127.0.0.1 --port 8080
   ```
2. Access the API documentation at `http://localhost:8080/docs`
3. Configure RTSP streams and detection parameters


