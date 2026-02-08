import cv2
import time
import os
import pickle
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =====================================================
# CONFIG
# =====================================================
SOURCE = 0
MODEL_PATH = "yolov8n.pt"
CONF = 0.5

CALIBRATION_MODE = True

OUTER_PAD_RATIO = 0.25
MIDDLE_SHRINK   = 0.20
INNER_SHRINK    = 0.40

# Côté "intérieur" de la salle sur l'image
INSIDE_SIDE = "right"  # "left" ou "right"

# ---- STOCKAGE (OPTIONNEL) ----
USE_ARCHIVE_REID = False  # toggle avec T

ARCHIVE_DIR  = "/home/teooff1700/AI_VP/AI/reid_archive"
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, "archive.pkl")

# IMPORTANT: plus strict + TTL réduit => évite "tout le monde = même personne"
ARCHIVE_TTL_SEC = 45
REID_SIM_TH = 0.20

# BBox min pour utiliser ReID (évite embeddings mauvais quand personne loin)
MIN_BBOX_AREA_FOR_REID = 6000  # ajuste 4000-12000 selon ta caméra

WINDOW_NAME = "People Counter (instant crossing + ReID safe)"
FULLSCREEN = True
WIN_W, WIN_H = 1280, 720

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

REARM_TIMEOUT = 2.0
STATE_TTL = 1.5

# =====================================================
# UTILS
# =====================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_archive_file():
    ensure_dir(ARCHIVE_DIR)
    with open(ARCHIVE_FILE, "wb") as f:
        pickle.dump([], f)

def save_archive(archive):
    with open(ARCHIVE_FILE, "wb") as f:
        pickle.dump(archive, f)

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def cosine_distance(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return 1.0 - float(np.dot(a, b))

def archive_cleanup(archive, now):
    return [(ts, emb, tag) for (ts, emb, tag) in archive if (now - ts) <= ARCHIVE_TTL_SEC]

def is_in_archive(archive, emb, now, tag):
    archive = archive_cleanup(archive, now)
    if emb is None:
        return False, archive
    emb = l2_normalize(emb)
    for _, old_emb, old_tag in archive:
        if old_tag != tag:
            continue
        if cosine_distance(emb, old_emb) <= REID_SIM_TH:
            return True, archive
    return False, archive

def add_to_archive(archive, emb, now, tag):
    archive = archive_cleanup(archive, now)
    if emb is None:
        return archive
    emb = l2_normalize(emb)
    archive.append((now, emb, tag))
    save_archive(archive)
    return archive

def centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def build_zones_from_door(x_left, x_right, w, h):
    x_left, x_right = sorted([int(x_left), int(x_right)])
    door_w = max(2, x_right - x_left)

    outer_pad = int(door_w * OUTER_PAD_RATIO)
    outer_x1 = clamp(x_left - outer_pad, 0, w - 2)
    outer_x2 = clamp(x_right + outer_pad, 1, w - 1)

    def shrink_segment(a, b, shrink_ratio):
        s = int(door_w * shrink_ratio)
        return clamp(a + s, 0, w - 2), clamp(b - s, 1, w - 1)

    mid_x1, mid_x2 = shrink_segment(x_left, x_right, MIDDLE_SHRINK)
    in_x1, in_x2   = shrink_segment(x_left, x_right, INNER_SHRINK)

    if mid_x2 <= mid_x1 + 2:
        mid_x1, mid_x2 = x_left, x_right
    if in_x2 <= in_x1 + 2:
        in_x1, in_x2 = mid_x1, mid_x2

    OUTER  = (outer_x1, 0, outer_x2, h - 1)
    MIDDLE = (mid_x1,   0, mid_x2,   h - 1)
    INNER  = (in_x1,    0, in_x2,    h - 1)
    return OUTER, MIDDLE, INNER

def draw_zones(frame, OUTER, MIDDLE, INNER):
    cv2.rectangle(frame, (OUTER[0], OUTER[1]), (OUTER[2], OUTER[3]), WHITE, 2)
    cv2.putText(frame, "OUTER", (OUTER[0] + 6, 30), FONT, 0.8, WHITE, 2)

    cv2.rectangle(frame, (MIDDLE[0], MIDDLE[1]), (MIDDLE[2], MIDDLE[3]), CYAN, 2)
    cv2.putText(frame, "MIDDLE", (MIDDLE[0] + 6, 60), FONT, 0.8, CYAN, 2)

    cv2.rectangle(frame, (INNER[0], INNER[1]), (INNER[2], INNER[3]), GREEN, 2)
    cv2.putText(frame, "INNER (INSTANT CROSS)", (INNER[0] + 6, 90), FONT, 0.8, GREEN, 2)

def side_of_inner(px, inner_rect):
    cx = (inner_rect[0] + inner_rect[2]) / 2
    return "left" if px < cx else "right"

# =====================================================
# CALIBRATION (mouse)
# =====================================================
class DoorCalibrator:
    def __init__(self):
        self.clicks = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < 2:
            self.clicks.append((x, y))

    def done(self):
        return len(self.clicks) == 2

    def get_xs(self):
        return self.clicks[0][0], self.clicks[1][0]

def run_calibration(cap):
    calib = DoorCalibrator()
    calib_win = "CALIBRATION: click LEFT then RIGHT edge of the door (ENTER confirm, R retry, Q quit)"

    cv2.namedWindow(calib_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(calib_win, calib.on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        disp = frame.copy()
        h, w = disp.shape[:2]

        cv2.putText(disp, "Calibration: click LEFT edge then RIGHT edge of the door",
                    (20, 40), FONT, 0.9, RED, 3)
        cv2.putText(disp, "ENTER=confirm  R=retry  Q=quit",
                    (20, 85), FONT, 0.8, RED, 2)

        for i, (x, y) in enumerate(calib.clicks):
            cv2.circle(disp, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(disp, f"{i+1}", (x + 10, y - 10), FONT, 0.8, (0, 255, 0), 2)

        if calib.done():
            x1, x2 = calib.get_xs()
            x_left, x_right = sorted([x1, x2])
            cv2.rectangle(disp, (x_left, 0), (x_right, h - 1), (0, 255, 255), 2)
            cv2.putText(disp, "Door selected", (x_left + 6, 120), FONT, 0.8, (0, 255, 255), 2)

        cv2.imshow(calib_win, disp)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            cv2.destroyWindow(calib_win)
            return None
        elif k == ord('r'):
            calib.clicks.clear()
        elif k in (13, 10):
            if calib.done():
                cv2.destroyWindow(calib_win)
                return calib.get_xs()

# =====================================================
# INIT
# =====================================================
reset_archive_file()
archive = []

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la caméra.")

door_x_left, door_x_right = 260, 380
if CALIBRATION_MODE:
    xs = run_calibration(cap)
    if xs is None:
        print("Calibration annulée. Fin.")
        cap.release()
        raise SystemExit(0)
    door_x_left, door_x_right = xs

model = YOLO(MODEL_PATH)

# DeepSort avec embedder + GPU (important sur Jetson)
tracker = DeepSort(
    max_age=35,
    n_init=1,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if FULLSCREEN:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)

tracks_state = {}
in_count = 0
out_count = 0

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    h, w = frame.shape[:2]

    OUTER, MIDDLE, INNER = build_zones_from_door(door_x_left, door_x_right, w, h)

    draw_zones(frame, OUTER, MIDDLE, INNER)
    cv2.putText(frame, f"IN: {in_count}", (20, 50), FONT, 1.2, RED, 4)
    cv2.putText(frame, f"OUT: {out_count}", (20, 100), FONT, 1.2, RED, 4)

    cv2.putText(frame, f"ReID archive: {'ON' if USE_ARCHIVE_REID else 'OFF'} (T)",
                (20, 150), FONT, 0.7, YELLOW, 2)
    cv2.putText(frame, "Q=quit  R=reset  C=recalibrate  T=toggle ReID",
                (20, h - 20), FONT, 0.7, WHITE, 2)

    # YOLO detection (person only)
    y = model(frame, conf=CONF, classes=[0], verbose=False)[0]
    detections = []
    if y.boxes:
        for b in y.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            conf_det = float(b.conf[0].cpu().numpy())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf_det, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        tid = t.track_id
        l, t_, r, b = map(int, t.to_ltrb())
        cx, cy = centroid(l, t_, r, b)

        if tid not in tracks_state:
            tracks_state[tid] = {
                "state": "none",
                "last_seen": now,
                "counted_time": 0.0,
                "enter_side": None,
                "last_side": None,
                "crossed": False,
                "last_dir": "",
                "last_result": ""
            }

        st = tracks_state[tid]
        st["last_seen"] = now

        in_outer  = point_in_rect(cx, cy, OUTER)
        in_middle = point_in_rect(cx, cy, MIDDLE)
        in_inner  = point_in_rect(cx, cy, INNER)

        cur_side = side_of_inner(cx, INNER)

        # --- robust embedding fetch ---
        emb = None
        if hasattr(t, "features") and t.features is not None and len(t.features) > 0:
            emb = t.features[-1]
        elif hasattr(t, "feature") and t.feature is not None:
            emb = t.feature

        # bbox area (for ReID quality guard)
        area = (r - l) * (b - t_)
        use_emb = (emb is not None) and (area >= MIN_BBOX_AREA_FOR_REID)

        # -------------------------
        # State machine (INSTANT CROSS)
        # -------------------------
        if st["state"] == "none":
            if in_outer:
                st["state"] = "outer"

        elif st["state"] == "outer":
            if in_middle:
                st["state"] = "middle"
            if not in_outer:
                st["state"] = "none"

        elif st["state"] == "middle":
            if in_inner:
                st["state"] = "inner"
                st["enter_side"] = cur_side
                st["last_side"] = cur_side
                st["crossed"] = False
            if not in_outer:
                st["state"] = "none"

        elif st["state"] == "inner":
            if in_inner:
                if st["last_side"] is None:
                    st["last_side"] = cur_side

                if (not st["crossed"]) and (cur_side != st["last_side"]):
                    exit_side = cur_side

                    # ✅ IN/OUT inversé comme tu veux
                    direction = "OUT" if exit_side == INSIDE_SIDE else "IN"

                    found = False
                    if USE_ARCHIVE_REID and use_emb:
                        found, archive = is_in_archive(archive, emb, now, direction)

                    if not found:
                        if direction == "IN":
                            in_count += 1
                        else:
                            out_count += 1

                        if USE_ARCHIVE_REID and use_emb:
                            archive = add_to_archive(archive, emb, now, direction)

                    st["last_dir"] = direction
                    st["last_result"] = "FOUND" if (USE_ARCHIVE_REID and found) else "NEW"

                    st["crossed"] = True
                    st["state"] = "counted"
                    st["counted_time"] = now

                st["last_side"] = cur_side
            else:
                st["state"] = "none"
                st["enter_side"] = None
                st["last_side"] = None
                st["crossed"] = False
                st["last_dir"] = ""
                st["last_result"] = ""

        elif st["state"] == "counted":
            if (not in_middle) or ((now - st.get("counted_time", now)) > REARM_TIMEOUT):
                st["state"] = "none"
                st["enter_side"] = None
                st["last_side"] = None
                st["crossed"] = False
                st["last_dir"] = ""
                st["last_result"] = ""

        # Draw bbox
        cv2.rectangle(frame, (l, t_), (r, b), BLUE, 2)
        info = f"ID {tid} [{st['state']}] area={area} side={cur_side} {st.get('last_dir','')} {st.get('last_result','')}"
        cv2.putText(frame, info, (l, max(20, t_ - 7)), FONT, 0.62, BLUE, 2)

    # Purge des tracks fantômes
    to_delete = [tid for tid, st in tracks_state.items()
                 if (now - st.get("last_seen", now)) > STATE_TTL]
    for tid in to_delete:
        del tracks_state[tid]

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        in_count = 0
        out_count = 0
        tracks_state.clear()
        archive.clear()
        reset_archive_file()
    elif key == ord('c'):
        xs = run_calibration(cap)
        if xs is not None:
            door_x_left, door_x_right = xs
    elif key == ord('t'):
        USE_ARCHIVE_REID = not USE_ARCHIVE_REID
        # important: on vide l’archive en switch ON/OFF
        archive.clear()
        reset_archive_file()

cap.release()
cv2.destroyAllWindows()
