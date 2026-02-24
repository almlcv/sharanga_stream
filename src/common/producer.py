import time
import multiprocessing as mp
from typing import Dict
import queue
from src.config.config import RTSPStream, DETECTION_INTERVAL
from src.common.stream_capture import StreamCapture

def stream_producer(stream_config: RTSPStream, global_index: int, local_index: int, stats_queue,
                   detection_queues: Dict, shutdown_event):
    """Producer process"""
    cap = None
    try:
        cap = StreamCapture(stream_config)
        cap.start()

        frame_count = 0
        reconnect_count = 0
        consecutive_failures = 0
        model_id = stream_config.model_id
        detection_queue = detection_queues.get(model_id)

        print(f"[*] Producer started: stream={global_index} name={stream_config.name}")

        while not shutdown_event.is_set():
            ret, frame = cap.read()

            if shutdown_event.is_set():
                break

            if not cap.is_alive():
                consecutive_failures = 30
                ret = False

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    if shutdown_event.is_set():
                        break

                    reconnect_count += 1

                    if cap:
                        try:
                            cap.release()
                        except:
                            pass

                    wait_time = min(2 ** min(reconnect_count - 1, 3), 10)
                    for _ in range(int(wait_time * 10)):
                        if shutdown_event.is_set():
                            return
                        time.sleep(0.1)

                    if shutdown_event.is_set():
                        return

                    cap = StreamCapture(stream_config)
                    cap.start()
                    consecutive_failures = 0

                continue

            consecutive_failures = 0
            frame_count += 1

            # Send frames for detection
            if frame_count % DETECTION_INTERVAL == 0 and detection_queue and not detection_queue.full():
                try:
                    detection_queue.put_nowait((local_index, stream_config.name, frame, time.time()))
                except:
                    pass

    finally:
        print(f"[*] Producer {global_index} exiting...")
        if cap:
            try:
                cap.release()
            except:
                pass