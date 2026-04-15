import multiprocessing as mp
from fastapi import FastAPI
from api.routes import detection
from fastapi.middleware.cors import CORSMiddleware

def configure_multiprocessing():
    """Configure multiprocessing for CUDA compatibility.

    Must be called before creating any Process objects.
    Uses spawn method to avoid fork-related issues with CUDA/NumPy.
    """
    try:
        current_method = mp.get_start_method(allow_none=True)

        if current_method is None:
            mp.set_start_method('spawn')
            print("[✓] Multiprocessing start method set to 'spawn'")
        elif current_method != 'spawn':
            print(f"[!] Warning: Multiprocessing method already set to '{current_method}'")
            print("[!] CUDA may not work properly. Restart application to fix.")
        else:
            print(f"[✓] Multiprocessing already configured with '{current_method}'")
    except RuntimeError as e:
        print(f"[!] Could not set multiprocessing method: {e}")


# Configure multiprocessing before creating any processes
if __name__ == "__main__":
    configure_multiprocessing()

description = """
# Multi-Streaming & Multi-Model Detection API

## Overview
This FastAPI service runs real-time YOLO detection pipelines over RTSP streams. It manages detector and producer worker processes per model and exposes simple REST endpoints to control and view detection output.

### Supported Models
- `fire_smoke`
- `ppe` 
- `load_unload`

### Key Endpoints
All endpoints use the prefix `/api/detection`:

- **POST** `/{model_name}/start` — Start background detection for a named model
- **POST** `/{model_name}/stop` — Stop detection for a named model
- **GET** `/{model_name}/stream` — MJPEG GUI stream for visual output (multipart/x-mixed-replace)

## Quick Examples

**Start Detection for the `fire_smoke` Model:**
- Localhost: `curl -X POST http://localhost:8080/api/detection/fire_smoke/start`
- External: `curl -X POST https://rabstream.alvision.in/api/detection/fire_smoke/start`

**Stop Detection:**
- Localhost: `curl -X POST http://localhost:8080/api/detection/fire_smoke/stop`
- External: `curl -X POST https://rabstream.alvision.in/api/detection/fire_smoke/stop`

**Check Status:**
- Localhost: `curl http://localhost:8080/api/detection/fire_smoke/status`
- External: `curl https://rabstream.alvision.in/api/detection/fire_smoke/status`

**View the MJPEG GUI Stream:**
- Localhost: `http://localhost:8080/api/detection/fire_smoke/stream`
- External: `https://rabstream.alvision.in/api/detection/fire_smoke/stream`

## Response Codes
- **200** — Success
- **400** — Bad Request (invalid model name or parameters)
- **404** — Model not found
- **500** — Internal server error
"""


app = FastAPI(title="Stream & Detection API", 
              description=description,
              docs_url="/docs",
              redoc_url="/redoc",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sharangaai.netlify.app",
                   "http://localhost:5173",
                   "http://192.168.1.111:5173", ],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(detection.router, prefix="/api")


@app.get("/", tags=["health"])
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Stream & Detection API",
        "version": "1.0.0",
    }
