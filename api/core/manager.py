import multiprocessing as mp
import signal
import os
import time
from datetime import datetime
from typing import Dict, List
from src.config.config import STREAM_CONFIGS, MODEL_CONFIGS, QUEUE_SIZE
from src.common.producer import stream_producer
from src.common.yolo_detector import yolo_detector_process

MODEL_MAPPING = {
    "fire_smoke": 0,
    "ppe": 1,
    "load_unload": 2,
}

active_processes: Dict[str, Dict] = {}


def get_mp_context():
    """Get spawn context for CUDA compatibility.

    Returns:
        Multiprocessing context configured for spawn method
    """
    try:
        return mp.get_context('spawn')
    except ValueError:
        # Fallback if spawn not available (shouldn't happen on modern Python)
        print("[!] Warning: spawn method not available, using default")
        return mp


def start_model_processes(model_name: str, model_id: int, visual_streaming: bool = False) -> List[Dict]:
    """Start detector and producers for a specific model.

    Args:
        model_name: Name of the model (fire_smoke, ppe, load_unload)
        model_id: Numeric ID of the model
        visual_streaming: Whether to enable visual streaming initially

    Returns:
        List of stream details that were started

    Raises:
        ValueError: If model is already running or no streams configured
    """
    if model_name in active_processes:
        if active_processes[model_name]["status"] == "running":
            raise ValueError(f"{model_name} is already running")

    # Get streams for this model
    model_streams = [s for s in STREAM_CONFIGS if s.model_id == model_id]
    if not model_streams:
        raise ValueError(f"No streams configured for model {model_name}")

    # Get spawn context
    ctx = get_mp_context()

    # Create queues using spawn context
    detection_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    display_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    shutdown_event = ctx.Event()

    # Get model config
    model_config = MODEL_CONFIGS[model_id]

    print(f"[*] Starting {model_name} with {len(model_streams)} streams...")

    # Start detector process using spawn context
    detector = ctx.Process(
        target=yolo_detector_process,
        args=(model_id, model_config, detection_queue, display_queue, shutdown_event),
        daemon=False,
        name=f"detector_{model_name}"
    )
    detector.start()
    print(f"[✓] Detector process started (PID: {detector.pid})")

    # Start producer processes for each stream
    producers = []
    stream_details = []

    for local_idx, stream in enumerate(model_streams):
        global_idx = STREAM_CONFIGS.index(stream)
        p = ctx.Process(
            target=stream_producer,
            args=(stream, global_idx, local_idx, None,  # No stats_queue
                  {model_id: detection_queue}, shutdown_event),
            daemon=False,
            name=f"producer_{model_name}_{local_idx}"
        )
        p.start()
        producers.append(p)

        stream_details.append({
            "index": local_idx,
            "name": stream.name,
            "channel": str(stream.channel),
            "url": "hidden"
        })
        print(f"[✓] Producer {local_idx} started (PID: {p.pid}) - {stream.name}")

    # Store process information
    active_processes[model_name] = {
        "model_id": model_id,
        "detector": detector,
        "producers": producers,
        "queues": {
            "detection": detection_queue,
            "display": display_queue,
        },
        "shutdown_event": shutdown_event,
        "status": "running",
        "stream_count": len(model_streams),
        "visual_streaming": visual_streaming,
        "streaming_stopped": False,
        "start_time": datetime.now(),
    }

    print(f"[✓] Started {model_name}: detector + {len(producers)} producers")
    return stream_details


def stop_model_processes(model_name: str) -> None:
    """Stop processes gracefully with timeout, then force terminate."""
    if model_name not in active_processes:
        print(f"[!] {model_name} not found")
        return
    
    state = active_processes[model_name]
    print(f"[*] Stopping {model_name}...")
    
    # Mark as stopping
    state["status"] = "stopping"
    state["visual_streaming"] = False
    
    # Signal shutdown
    try:
        shutdown_event = state.get("shutdown_event")
        if shutdown_event:
            shutdown_event.set()
            print(f"[✓] Shutdown signal sent")
    except Exception as e:
        print(f"[!] Error setting shutdown event: {e}")
    
    # Collect all processes
    all_processes = []
    if state.get("detector"):
        all_processes.append(("detector", state["detector"]))
    
    for idx, proc in enumerate(state.get("producers", [])):
        if proc:
            all_processes.append((f"producer_{idx}", proc))
    
    if not all_processes:
        print(f"[!] No processes to stop for {model_name}")
        del active_processes[model_name]
        return
    
    # Phase 1: Try graceful shutdown (3 second timeout)
    print(f"[*] Phase 1: Waiting for {len(all_processes)} processes to exit gracefully (3s)...")
    graceful_timeout = time.time() + 3.0
    
    for name, proc in all_processes:
        if proc and proc.is_alive():
            remaining = max(0.1, graceful_timeout - time.time())
            proc.join(timeout=remaining)
            if not proc.is_alive():
                print(f"[✓] {name} exited gracefully")
            else:
                print(f"[!] {name} did not exit gracefully")
    
    # Phase 2: Terminate remaining processes (SIGTERM)
    still_alive = [(name, proc) for name, proc in all_processes if proc and proc.is_alive()]
    
    if still_alive:
        print(f"[*] Phase 2: Terminating {len(still_alive)} remaining processes (SIGTERM)...")
        for name, proc in still_alive:
            try:
                proc.terminate()
                print(f"[*] Sent SIGTERM to {name} (PID: {proc.pid})")
            except Exception as e:
                print(f"[!] Error terminating {name}: {e}")
        
        # Wait for terminate to take effect
        terminate_timeout = time.time() + 2.0
        for name, proc in still_alive:
            if proc.is_alive():
                remaining = max(0.1, terminate_timeout - time.time())
                proc.join(timeout=remaining)
                if not proc.is_alive():
                    print(f"[✓] {name} terminated")
    
    # Phase 3: Kill remaining processes (SIGKILL)
    still_alive = [(name, proc) for name, proc in all_processes if proc and proc.is_alive()]
    
    if still_alive:
        print(f"[*] Phase 3: Killing {len(still_alive)} stubborn processes (SIGKILL)...")
        for name, proc in still_alive:
            try:
                os.kill(proc.pid, signal.SIGKILL)
                proc.join(timeout=1.0)
                print(f"[!] Killed {name} with SIGKILL (PID: {proc.pid})")
            except ProcessLookupError:
                print(f"[✓] {name} already gone")
            except Exception as e:
                print(f"[!] Could not kill {name}: {e}")
    
    # Phase 4: ABANDON queues (don't try to clean them)
    # With spawn method and forcefully killed processes, trying to clean
    # queues can cause deadlocks. Just abandon them and let Python GC handle it.
    print(f"[*] Phase 4: Abandoning queues...")
    
    for queue_name, queue in state.get("queues", {}).items():
        if queue:
            try:
                # Cancel the background thread immediately (non-blocking)
                queue.cancel_join_thread()
                print(f"[✓] {queue_name} queue abandoned")
            except Exception as e:
                print(f"[!] Error abandoning {queue_name} queue: {e}")
                # Continue anyway - don't block on queue errors
    
    # Remove from active processes immediately
    del active_processes[model_name]
    print(f"[✓] {model_name} stopped successfully")




def enable_visual_streaming(model_name: str) -> None:
    """Enable visual streaming for a model.

    Args:
        model_name: Name of the model

    Raises:
        ValueError: If model is not running
    """
    if model_name not in active_processes:
        raise ValueError(f"{model_name} not running")

    active_processes[model_name]["visual_streaming"] = True
    active_processes[model_name]["streaming_stopped"] = False
    print(f"[✓] Visual streaming enabled for {model_name}")


def stop_visual_streaming(model_name: str) -> None:
    """Stop visual streaming for a model (detection continues).

    Args:
        model_name: Name of the model

    Raises:
        ValueError: If model is not running
    """
    if model_name not in active_processes:
        raise ValueError(f"{model_name} not running")

    state = active_processes[model_name]
    state["visual_streaming"] = False
    state["streaming_stopped"] = True
    print(f"[✓] Visual streaming stopped for {model_name}")


def get_system_status() -> Dict:
    """Get status of all running models.

    Returns:
        Dictionary containing system status
    """
    models_status = {}

    for model_name, state in active_processes.items():
        uptime = None
        if state.get("start_time"):
            uptime = (datetime.now() - state["start_time"]).total_seconds()

        # Check process health
        detector_alive = state.get("detector") and state["detector"].is_alive()
        producers_alive = sum(1 for p in state.get("producers", []) if p and p.is_alive())

        models_status[model_name] = {
            "status": state.get("status"),
            "model_id": state.get("model_id"),
            "streams": state.get("stream_count", 0),
            "visual_streaming": state.get("visual_streaming", False),
            "uptime_seconds": round(uptime, 2) if uptime else None,
            "detector_alive": detector_alive,
            "producers_alive": producers_alive,
            "producers_total": len(state.get("producers", [])),
        }

    return {
        "status": "healthy",
        "active_models": len(active_processes),
        "multiprocessing_method": mp.get_start_method(),
        "models": models_status,
    }


def cleanup_all_processes() -> None:
    """Stop all running models during application shutdown."""
    print("[*] Cleaning up all processes...")

    model_names = list(active_processes.keys())
    for model_name in model_names:
        try:
            stop_model_processes(model_name)
        except Exception as e:
            print(f"[!] Error stopping {model_name}: {e}")

    print("[✓] All processes cleaned up")