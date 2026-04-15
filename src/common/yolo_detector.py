import multiprocessing as mp
import os
from src.config.config import ModelConfig

try:
    from src.scripts.fire_smoke import process_detection as process_fire_smoke
    from src.scripts.ppe import process_detection as process_ppe
    from src.scripts.load_unload import process_detection as process_load_unload
except ImportError:
    def process_fire_smoke(frame, results, model_id: int, stream_index: int) -> bool:
        return False
    def process_ppe(frame, results, model_id: int, stream_index: int) -> bool:
        return False
    def process_load_unload(frame, results, model_id: int, stream_index: int) -> bool:
        return False

MODEL_PROCESSORS = {
    0: process_fire_smoke,
    1: process_ppe,
    2: process_load_unload,
}

# Models that require tracking
TRACKING_MODELS = {2}  # Model 2 (load_unload) uses tracking

def yolo_detector_process(model_id: int, model_config: ModelConfig, detection_queue: mp.Queue,
                         result_queue_or_display: mp.Queue, 
                         display_queue_or_event=None, 
                         shutdown_event=None):
    """
    Detector with BACKWARD COMPATIBILITY for both old and new manager.py
    """
    # Auto-detect which signature was used
    if shutdown_event is None:
        display_queue = result_queue_or_display
        shutdown_event = display_queue_or_event
        print(f"[*] Loading YOLO model {model_id} (NEW signature)")
    else:
        display_queue = display_queue_or_event
        print(f"[*] Loading YOLO model {model_id} (OLD signature - result_queue ignored)")

    # CUDA / cuDNN Initialization fixes
    try:
        import torch
        from ultralytics import YOLO
        
        # Set environment variables for better 40-series support and to avoid cuDNN issues
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        
        if torch.cuda.is_available():
            # Select specific GPU
            torch.cuda.set_device(model_config.gpu_id)
            
            # Disable cuDNN as a workaround for CUDNN_STATUS_NOT_INITIALIZED
            # On an RTX 4090, standard CUDA kernels are still extremely fast.
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            
            # Enable TF32 for better performance on Ampere/Ada GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear cache before loading model
            torch.cuda.empty_cache()
            print(f"[✓] CUDA initialized (cuDNN disabled) for model {model_id} on GPU {model_config.gpu_id}")
        else:
            print(f"[!] CUDA NOT AVAILABLE for model {model_id}")

        model = YOLO(model_config.model_path)
        if model_id in TRACKING_MODELS:
            print(f"[✓] Model {model_id} loaded with TRACKING enabled")
        else:
            print(f"[✓] Model {model_id} loaded successfully")
    except Exception as e:
        print(f"[!] Failed to load model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return

    processor = MODEL_PROCESSORS.get(model_id)
    use_tracking = model_id in TRACKING_MODELS

    try:
        while shutdown_event is not None and not shutdown_event.is_set():
            try:
                index, stream_name, frame, timestamp = detection_queue.get(timeout=0.5)
            except:
                if shutdown_event.is_set():
                    break
                continue

            if shutdown_event.is_set():
                break

            try:
                # Use tracking or regular detection based on model configuration
                if use_tracking:
                    results = model.track(
                        frame,
                        conf=model_config.conf_threshold,
                        iou=model_config.iou_threshold,
                        classes=model_config.classes,
                        batch=model_config.batch_size,
                        device=model_config.gpu_id,
                        tracker='botsort.yaml',
                        seed=42,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = model(
                        frame,
                        conf=model_config.conf_threshold,
                        iou=model_config.iou_threshold,
                        classes=model_config.classes,
                        batch=model_config.batch_size,
                        device=model_config.gpu_id,
                        seed=42,
                        verbose=False,
                    )

                # Process detections if processor exists
                if processor:
                    try:
                        processor(frame, results, model_id, index, stream_name)
                    except Exception as e:
                        print(f"[!] Model {model_id} processing error: {e}")

                # Update display queue (drop old frames if full)
                if display_queue is not None:
                    try:
                        while display_queue.qsize() > 3:
                            try:
                                display_queue.get_nowait()
                            except:
                                break
                        display_queue.put_nowait((index, frame, timestamp))
                    except Exception:
                        pass
            
            except Exception as e:
                # Catch detection errors specifically to print more info
                print(f"[!] Model {model_id} detection loop error: {e}")
                if "cuDNN" in str(e):
                    print("[*] Attempting to recover from cuDNN error...")
                    torch.cuda.empty_cache()
                # Optionally break or continue based on severity
                # For cuDNN initialization errors, continuing might just spam
                if "NOT_INITIALIZED" in str(e):
                    break

    except KeyboardInterrupt:
        print(f"[*] Model {model_id} detector interrupted")
    except Exception as e:
        print(f"[!] Model {model_id} process fatal error: {e}")
    finally:
        print(f"[✓] Model {model_id} detector stopped cleanly")