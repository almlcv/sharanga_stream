import cv2
import asyncio
import time
import math
import numpy as np
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from api.core.manager import active_processes  # Remove enable_visual_streaming import
from src.config.config import STREAM_WIDTH, STREAM_HEIGHT

# TurboJPEG: 2-3x faster JPEG encoding than cv2.imencode
try:
    from turbojpeg import TurboJPEG
    _tj = TurboJPEG('/home/aiserver/miniconda3/pkgs/libjpeg-turbo-2.0.0-h9bf148f_0/lib/libturbojpeg.so')
    print('[✓] TurboJPEG loaded — fast JPEG encoding enabled')
except Exception as e:
    _tj = None
    print(f'[!] TurboJPEG not available, using OpenCV fallback: {e}')


def encode_jpeg(frame: np.ndarray, quality: int = 65) -> bytes:
    """Encode frame to JPEG using TurboJPEG (fast) or OpenCV (fallback)."""
    if _tj is not None:
        return _tj.encode(frame, quality=quality)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


async def gui_stream(model_name: str):
    """MJPEG GUI stream - displays annotated frames from detector.
    
    Note: Assumes visual_streaming is already enabled by the caller.
    """
    from api.core.manager import MODEL_MAPPING
    
    if model_name not in active_processes or active_processes[model_name]["status"] != "running":
        raise HTTPException(status_code=400, detail=f"{model_name} not running")
    
    # Check if streaming was explicitly stopped
    if active_processes[model_name].get("streaming_stopped", False):
        raise HTTPException(
            status_code=400, 
            detail=f"Streaming disabled for {model_name}. Access /stream again to enable."
        )
    
    # Verify streaming is enabled (should be set by caller)
    if not active_processes[model_name].get("visual_streaming"):
        raise HTTPException(
            status_code=400,
            detail=f"Visual streaming not enabled for {model_name}"
        )
    
    num_streams = active_processes[model_name].get("stream_count", 1)
    cols = math.ceil(math.sqrt(num_streams))
    rows = math.ceil(num_streams / cols)
    grid_width = STREAM_WIDTH * cols
    grid_height = STREAM_HEIGHT * rows
    
    async def frame_generator():
        last_frame_time = {}
        stream_frames = {}
        last_grid_update = time.time()
        no_frames_logged = False
        
        try:
            while True:
                # Always fetch fresh state
                if model_name not in active_processes:
                    print(f"[!] Model {model_name} no longer active, stopping stream")
                    break
                
                current_state = active_processes[model_name]
                
                # Check if streaming was stopped
                if current_state.get("streaming_stopped", False):
                    print(f"[!] Streaming explicitly stopped for {model_name}")
                    break
                
                if not current_state.get("visual_streaming", False):
                    print(f"[!] Visual streaming disabled for {model_name}")
                    break
                
                if current_state.get("status") != "running":
                    print(f"[!] Model {model_name} not running, stopping stream")
                    break
                
                display_queue = current_state["queues"].get("display")
                if display_queue is None:
                    await asyncio.sleep(0.1)
                    continue
                
                frames_updated = False
                processed = 0
                
                while not display_queue.empty() and processed < 20:
                    try:
                        idx, frame, timestamp = display_queue.get_nowait()
                        stream_frames[idx] = frame
                        last_frame_time[idx] = time.time()
                        frames_updated = True
                        processed += 1
                        no_frames_logged = False
                    except Exception:
                        break
                
                current_time = time.time()
                
                if frames_updated or (current_time - last_grid_update >= 0.066):
                    canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                    stream_indices = sorted(stream_frames.keys())
                    
                    if not stream_indices:
                        if not no_frames_logged:
                            print(f"[DEBUG] No frames available yet for {model_name}")
                            no_frames_logged = True
                        await asyncio.sleep(0.05)
                        continue
                    
                    for grid_idx, actual_idx in enumerate(stream_indices):
                        row, col = divmod(grid_idx, cols)
                        y1, y2 = row * STREAM_HEIGHT, (row + 1) * STREAM_HEIGHT
                        x1, x2 = col * STREAM_WIDTH, (col + 1) * STREAM_WIDTH
                        
                        if current_time - last_frame_time.get(actual_idx, 0) < 10:
                            frame = stream_frames[actual_idx]
                            canvas[y1:y2, x1:x2] = frame
                    
                    jpeg_bytes = encode_jpeg(canvas, quality=65)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                    
                    last_grid_update = current_time
                else:
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print(f"[*] MJPEG stream cancelled for {model_name}")
        except Exception as e:
            print(f"[!] MJPEG stream error ({model_name}): {e}")
        finally:
            print(f"[*] MJPEG stream ended for {model_name}")
            stream_frames.clear()
            last_frame_time.clear()
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )
