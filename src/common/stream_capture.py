import subprocess
import numpy as np
import signal
import os
import threading
import queue
from src.config.config import RTSPStream, STREAM_WIDTH, STREAM_HEIGHT

class StreamCapture:
    """
    Capture video from RTSP stream using FFmpeg with threaded buffer management.
    Solves 'rubber banding' by ensuring the pipe is continuously drained.
    """

    def __init__(self, stream_config: RTSPStream):
        self.config = stream_config
        self.process = None
        self.frame_size = STREAM_WIDTH * STREAM_HEIGHT * 3
        # Threading primitives
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.read_thread = None

    def start(self):
        """Start the FFmpeg process and the reader thread"""
        command = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-stimeout', '10000000',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-i', self.config.url,
            '-vf', f'scale_cuda={STREAM_WIDTH}:{STREAM_HEIGHT},hwdownload,format=nv12,format=bgr24',
            '-f', 'image2pipe',
            '-vcodec', 'rawvideo',
            '-fflags', 'nobuffer+flush_packets',
            '-flags', 'low_delay',
            'pipe:1'
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=100,
                preexec_fn=os.setsid
            )

            self.stop_event.clear()
            self.read_thread = threading.Thread(target=self._reader_worker, daemon=True)
            self.read_thread.start()
        except Exception as e:
            print(f"[!] Failed to start FFmpeg for {self.config.name}: {e}")
            self.release()

    def _reader_worker(self):
        """
        Background thread:
        1. Reads raw bytes from FFmpeg
        2. Converts to numpy array
        3. Puts in queue (dropping old frames if necessary)
        """
        while not self.stop_event.is_set():
            if self.process is None or self.process.stdout is None:
                break

            try:
                raw_bytes = self.process.stdout.read(self.frame_size)

                if len(raw_bytes) != self.frame_size:
                    break

                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((STREAM_HEIGHT, STREAM_WIDTH, 3))

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Drop old frame
                    except queue.Empty:
                        pass

                self.frame_queue.put(frame)
            except Exception:
                break

        self.stop_event.set()

    def is_alive(self):
        """Check if both the process and the reader thread are healthy"""
        process_alive = self.process and self.process.poll() is None
        thread_alive = self.read_thread and self.read_thread.is_alive()
        return process_alive and thread_alive and not self.stop_event.is_set()

    def read(self):
        """
        Get the latest frame.
        Returns: (bool, frame)
        """
        if not self.is_alive():
            return False, None

        try:
            frame = self.frame_queue.get(timeout=0.2)
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        """Stop threads and kill FFmpeg process"""
        self.stop_event.set()

        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception:
                pass

            try:
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except Exception:
                pass

        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

        self.process = None
        self.read_thread = None