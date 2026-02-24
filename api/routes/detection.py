from fastapi import APIRouter, HTTPException
from api.models.responses import StartResponse, StopResponse, StatusResponse
from api.core.manager import (
    start_model_processes,
    stop_model_processes,
    # get_model_stats,
    enable_visual_streaming,
    stop_visual_streaming,
    MODEL_MAPPING,
    active_processes,
)
from src.common.gui_monitor import gui_stream

# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)

router = APIRouter(prefix="/detection", tags=["Unified Detection"])


# -------------------------------------------------------------------------
# Generic Endpoints
# -------------------------------------------------------------------------

@router.post(
    "/{model_name}/start",
    response_model=StartResponse,
    summary="Start detection for a model",
    description=(
        "Start background detection processes (detector + stream producers) for the given "
        "model_name. On success returns a `StartResponse` with the number of streams started "
        "and details for each stream. Possible errors: 400 if model is already running, "
        "404 if the model name is unknown or no streams are configured for the model."
    ),
    responses={
        200: {
            "description": "Detection started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "fire_smoke detection started",
                        "model_name": "fire_smoke",
                        "model_id": 0,
                        "streams_started": 2,
                        "stream_details": [
                            {"index": 0, "name": "cam1", "channel": "1", "url": "hidden"},
                            {"index": 1, "name": "cam2", "channel": "2", "url": "hidden"}
                        ]
                    }
                }
            }
        },
        400: {"description": "Model is already running"},
        404: {"description": "Unknown model or no streams configured"},
    },
)
async def start_detection(model_name: str):
    """
    Start detection for a specific model.
    Can be called multiple times after stopping.
    Does NOT affect other running models.
    """
    if model_name not in MODEL_MAPPING:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    model_id = MODEL_MAPPING[model_name]
    
    try:
        stream_details = start_model_processes(model_name, model_id, visual_streaming=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StartResponse(
        status="success",
        message=f"{model_name} detection started",
        model_name=model_name,
        model_id=model_id,
        streams_started=len(stream_details),
        stream_details=stream_details,
    )


@router.post(
    "/{model_name}/stop",
    response_model=StopResponse,
    summary="Stop detection for a model",
    description=(
        "Stop background detection processes for the given model_name. This will attempt to "
        "terminate detector and producer processes and clear the runtime state for the model. "
        "Returns a `StopResponse` on success. 404 is returned if the model is not active/known."
    ),
    responses={
        200: {
            "description": "Detection stopped successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "fire_smoke detection stopped successfully",
                        "model_name": "fire_smoke"
                    }
                }
            }
        },
        404: {"description": "Unknown model or not running"},
    },
)
async def stop_detection(model_name: str):
    """
    Stop detection for any model.

    Path parameters:
    - model_name: name of the model to stop

    Returns:
    - StopResponse on success
    """
    if model_name not in active_processes:
        raise HTTPException(status_code=404, detail=f"{model_name} not running")

    try:
        stop_model_processes(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping {model_name}: {str(e)}")
    
    return StopResponse(
        status="success",
        message=f"{model_name} detection stopped successfully",
        model_name=model_name,
    )


# -------------------------------------------------------------------------
# GUI Stream Endpoint 
# -------------------------------------------------------------------------
@router.get(
    "/{model_name}/stream",
    summary="MJPEG GUI stream for a model",
    description=(
        "Return an MJPEG (multipart/x-mixed-replace) visual stream for the specified model. "
        "The model must be running and visual streaming enabled. Clients can render this URL "
        "directly in an <img> tag or an MJPEG-capable viewer. Returns 404 if the model is unknown "
        "or not running."
    ),
    response_description="MJPEG multipart stream",
    responses={
        200: {
            "description": "MJPEG stream",
            "content": {
                "multipart/x-mixed-replace;boundary=frame": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        404: {"description": "Unknown or non-running model"},
    },
)
async def detection_stream(model_name: str):
    """
    MJPEG GUI stream for any detection model.

    Path parameters:
    - model_name: name of the model whose visual stream to return

    Returns a streaming response with content-type multipart/x-mixed-replace.
    """
    if model_name not in active_processes:
        raise HTTPException(status_code=404, detail=f"{model_name} not running. Start it first with POST /{model_name}/start")
    
    if active_processes[model_name].get("status") != "running":
        raise HTTPException(status_code=400, detail=f"{model_name} is not in running state")
    
    # Automatically enable streaming when accessing this endpoint
    try:
        enable_visual_streaming(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enabling streaming: {str(e)}")
    
    return await gui_stream(model_name)

# @router.post("/{model_name}/enable_view")
# async def enable_view(model_name: str):
#     """Enable visual streaming for any model."""
#     enable_visual_streaming(model_name)
#     return {"status": "success", "message": f"Visual streaming enabled for {model_name}"}


@router.post("/{model_name}/stop_view")
async def stop_view(model_name: str):
    """`DO NOT USE THIS API - STILL UNDER DEVELOPMENT`"""
    """Stop visual streaming (but keep detection running)."""
    if model_name not in active_processes:
        raise HTTPException(status_code=404, detail=f"{model_name} not running")
    
    try:
        stop_visual_streaming(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping view: {str(e)}")
    
    return {
        "status": "success", 
        "message": f"Visual streaming stopped for {model_name}. Detection still running. Access /stream to resume viewing."
    }

# @router.get("/{model_name}/status", response_model=StatusResponse)
# async def status_detection(model_name: str):
#     """Get detection status for any model."""
#     if model_name not in MODEL_MAPPING:
#         raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

#     model_id = MODEL_MAPPING[model_name]
#     stats = get_model_stats(model_name)

#     return StatusResponse(
#         model_name=model_name,
#         model_id=model_id,
#         status=active_processes[model_name]["status"],
#         streams_running=active_processes[model_name].get("stream_count", 0),
#         uptime=stats.get("uptime"),
#         stats=stats,
#     )
