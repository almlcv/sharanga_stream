import cv2
import numpy as np
import asyncio
import inspect
import traceback
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
import atexit
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.common.utils.alert import send_alert_email
from matplotlib.path import Path as MplPath
from src.common.utils.db import db
from collections import defaultdict

COLLECTION_NAME = "PPEViolations"
collection = db[COLLECTION_NAME]

# ---------------- CONFIG ----------------
BASE_SAVE_DIR = "detected_frames/ppe"
Path(BASE_SAVE_DIR).mkdir(parents=True, exist_ok=True)

MODEL_PERSON = "person"
MODEL_HELMET = "helmet"
MODEL_NO_HELMET = "no-helmet"
ALERT_COOLDOWN = 300
COLOR_SAFE = (0, 225, 64)
COLOR_UNSAFE = (0,0,255)
COLOR_ROI = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

ROI_COORDS = "71,254,126,159,188,159,259,161,329,167,393,180,417,284,344,276,263,267,184,261"
frame_shape = (640, 640)
SAVE_WORKERS = 3
EMAIL_QUEUE_MAXSIZE = 500

# ---------- Logging ----------
import logging
logger = logging.getLogger("ppe_detector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ---------- Initialization ----------
def parse_roi_coords(coords_str: str) -> np.ndarray:
    coords_list = [int(x) for x in coords_str.split(',')]
    pts = [[coords_list[i], coords_list[i + 1]] for i in range(0, len(coords_list), 2)]
    return np.array(pts, dtype=np.int32)

ROI_POLYGON = parse_roi_coords(ROI_COORDS)
scaled_roi = ROI_POLYGON.copy()
roi_path = MplPath(scaled_roi)

_last_alert_time: Dict[int, datetime] = {}
_alert_locks: Dict[int, Lock] = {}
_alert_lock_mutex = Lock()
_email_queue: "Queue[Tuple[str, int]]" = Queue(maxsize=EMAIL_QUEUE_MAXSIZE)
_stop_event = Event()
_save_executor = ThreadPoolExecutor(max_workers=SAVE_WORKERS)

# Consecutive unsafe-frame counter
unsafe_counter = defaultdict(int)


# ---------- Drawing ----------
def draw_person_box(frame: np.ndarray, bbox: List[int], safe: bool, thickness=2):
    x1, y1, x2, y2 = bbox
    color = COLOR_SAFE if safe else COLOR_UNSAFE

    w = x2 - x1
    h = y2 - y1

    h_line = max(5, min(30, int(w * 0.3)))
    v_line = max(5, min(40, int(h * 0.3)))

    cv2.line(frame, (x1, y1), (x1 + h_line, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + v_line), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - h_line, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + v_line), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + h_line, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - v_line), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - h_line, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - v_line), color, thickness)


def draw_roi(frame, polygon):
    cv2.polylines(frame, [polygon], True, COLOR_ROI, 1, cv2.LINE_AA)


# ---------- Email Worker ----------
def email_worker_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _worker():
        while not _stop_event.is_set() or not _email_queue.empty():
            try:
                img_path, stream_name = _email_queue.get(timeout=0.5)
            except Empty:
                await asyncio.sleep(0.1)
                continue

            try:
                if inspect.iscoroutinefunction(send_alert_email):
                    await send_alert_email(img_path, "ppe", str(stream_name))
                else:
                    await loop.run_in_executor(None, send_alert_email, img_path, str(stream_name))
            finally:
                try:
                    _email_queue.task_done()
                except:
                    pass

    try:
        loop.run_until_complete(_worker())
    finally:
        loop.close()

_thread_email = Thread(target=email_worker_loop, daemon=False)
_thread_email.start()


# ---------- Shutdown ----------
def shutdown():
    _stop_event.set()
    _save_executor.shutdown(wait=True)
    try:
        _email_queue.join()
    except:
        pass
    _thread_email.join(timeout=10)

atexit.register(shutdown)


# ---------- Utility ----------
def get_alert_lock(stream_index: int) -> Lock:
    with _alert_lock_mutex:
        if stream_index not in _alert_locks:
            _alert_locks[stream_index] = Lock()
        return _alert_locks[stream_index]

def can_send_and_update_alert(stream_index: int) -> bool:
    lock = get_alert_lock(stream_index)
    with lock:
        now = datetime.now()
        last = _last_alert_time.get(stream_index)
        if last is None or (now - last).total_seconds() >= ALERT_COOLDOWN:
            _last_alert_time[stream_index] = now
            return True
        return False

def make_save_filename(stream_index: int) -> Optional[str]:
    try:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path(BASE_SAVE_DIR) / date_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return str(save_dir / f"violation_stream{stream_index}_{ts}.jpg")
    except:
        traceback.print_exc()
        return None


# ---------- ROI Helpers ----------
def lower_10pct_bottom_corners(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y2], [x2, y2]], dtype=float)

def is_box_in_roi(box: np.ndarray, roi_path: MplPath) -> bool:
    bl, br = lower_10pct_bottom_corners(box)
    return roi_path.contains_point(bl) and roi_path.contains_point(br)


def helmet_in_upper_40(person_box: np.ndarray, helmet_boxes: np.ndarray) -> np.ndarray:
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    if ph <= 0 or helmet_boxes.shape[0] == 0:
        return np.zeros((helmet_boxes.shape[0],), dtype=bool)

    upper_y2 = py1 + int(ph * 0.4)
    results = []

    for hb in helmet_boxes:
        hx1, hy1, hx2, hy2 = hb.astype(int)
        cx = (hx1 + hx2) / 2.0
        cy = (hy1 + hy2) / 2.0

        if not (px1 <= cx <= px2 and py1 <= cy <= upper_y2):
            results.append(False)
            continue

        inter_x1 = max(px1, hx1)
        inter_y1 = max(py1, hy1)
        inter_x2 = min(px2, hx2)
        inter_y2 = min(py2, hy2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            results.append(False)
            continue

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        p_area = (px2 - px1) * (py2 - py1)
        h_area = max(1, (hx2 - hx1) * (hy2 - hy1))

        iou = inter_area / float(p_area + h_area - inter_area)
        results.append(iou >= 0.02)

    return np.array(results, dtype=bool)


# ---------- Box Extractor ----------
def extract_boxes_array(boxes_obj) -> np.ndarray:
    try:
        if boxes_obj is None:
            return np.zeros((0, 6), dtype=float)

        if hasattr(boxes_obj, "xyxy") and hasattr(boxes_obj, "cls"):
            xy = boxes_obj.xyxy
            arr_xy = np.array(xy.cpu() if hasattr(xy, "cpu") else xy)
            cls_arr = np.array(boxes_obj.cls.cpu() if hasattr(boxes_obj.cls, "cpu") else boxes_obj.cls)
            conf_arr = np.array(boxes_obj.conf.cpu() if hasattr(boxes_obj.conf, "cpu") else boxes_obj.conf)
            if arr_xy.size == 0:
                return np.zeros((0, 6), dtype=float)

            out = np.zeros((arr_xy.shape[0], 6), dtype=float)
            out[:, :4] = arr_xy[:, :4]
            out[:, 4] = cls_arr
            out[:, 5] = conf_arr
            return out

        # fallback
        rows = []
        for b in boxes_obj:
            xyv = np.array(b[:4], dtype=float)
            cls = int(b[4]) if len(b) > 4 else 0
            conf = float(b[5]) if len(b) > 5 else 0.0
            rows.append([xyv[0], xyv[1], xyv[2], xyv[3], cls, conf])
        return np.array(rows, dtype=float)

    except:
        traceback.print_exc()
        return np.zeros((0, 6), dtype=float)


# ---------- Core Processing ----------
def process_detection(frame: np.ndarray, results, model_id: int, stream_index: int = 0, stream_name: str ="") -> bool:
    sent_alert = False

    if results is None:
        draw_roi(frame, scaled_roi)
        return False

    try:
        boxes_obj = results[0].boxes
    except:
        boxes_obj = getattr(results, "boxes", None)
        if boxes_obj is None:
            draw_roi(frame, scaled_roi)
            return False

    boxes_arr = extract_boxes_array(boxes_obj)
    if boxes_arr.shape[0] == 0:
        draw_roi(frame, scaled_roi)
        return False

    names = getattr(results[0], "names", None)
    labels = []

    for row in boxes_arr:
        cls_idx = int(row[4])
        label = None
        if names is not None:
            try:
                label = names[cls_idx]
            except:
                try:
                    label = names.get(cls_idx)
                except:
                    label = None
        labels.append(label)

    labels = np.array(labels, dtype=object)

    person_mask = labels == MODEL_PERSON
    helmet_mask = labels == MODEL_HELMET
    nohelmet_mask = labels == MODEL_NO_HELMET

    persons = boxes_arr[person_mask][:, :6]
    helmets = boxes_arr[helmet_mask][:, :4]
    nohelms = boxes_arr[nohelmet_mask][:, :4]

    if persons.shape[0] == 0:
        draw_roi(frame, scaled_roi)
        unsafe_counter[stream_index] = 0
        return False

    persons_in_roi_indices = [
        i for i, p in enumerate(persons)
        if is_box_in_roi(p[:4].astype(int), roi_path)
    ]

    helmets_arr = helmets.reshape(-1, 4) if helmets.size else np.zeros((0, 4))
    nohelms_arr = nohelms.reshape(-1, 4) if nohelms.size else np.zeros((0, 4))

    unsafe_persons = []

    for idx in persons_in_roi_indices:
        p = persons[idx]; pbox = p[:4].astype(int)

        has_helmet = np.any(helmet_in_upper_40(pbox, helmets_arr)) if helmets_arr.size else False
        has_nohelmet = np.any(helmet_in_upper_40(pbox, nohelms_arr)) if nohelms_arr.size else False
        is_safe = has_helmet and not has_nohelmet

        draw_person_box(frame, pbox.tolist(), safe=bool(is_safe))

        if not is_safe:
            unsafe_persons.append(idx)

    # ---------- 🔥 40-FRAME CONSECTUVE LOGIC ----------
    if unsafe_persons:
        unsafe_counter[stream_index] += 1
    else:
        unsafe_counter[stream_index] = 0

    # Alert only if 40 consecutive unsafe frames
    if unsafe_counter[stream_index] >= 40:
        unsafe_counter[stream_index] = 0  # reset after trigger

        if can_send_and_update_alert(stream_index):
            frame_copy = frame.copy()
            fname = make_save_filename(stream_index)

            if fname is not None:
                def _save_and_enqueue(path, img, sname):
                    try:
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        if cv2.imwrite(path, img):
                            try:
                                _email_queue.put((path, sname), timeout=2)
                            except Full:
                                logger.warning(f"Email queue full, dropping alert: {path}")
                            logger.info(f"[ALERT QUEUED] {path}")
                        else:
                            logger.error(f"[SAVE FAILED] {path}")
                    except:
                        traceback.print_exc()

                _save_executor.submit(_save_and_enqueue, fname, frame_copy, stream_name)
                sent_alert = True
                abs_fname = str(Path(fname).resolve())

                try:
                    record = {
                        "timestamp": datetime.now(),
                        "stream_name": stream_name,
                        "image_path": abs_fname,
                        "voilation_count": len(unsafe_persons),
                    }
                    collection.insert_one(record)
                except Exception as e:
                    logger.error(f"[DB ERROR] Could not insert violation record: {e}")

    draw_roi(frame, scaled_roi)
    return sent_alert
