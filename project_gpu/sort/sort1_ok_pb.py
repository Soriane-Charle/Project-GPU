import cv2
import numpy as np
from sort import Sort
 
# ================= CONFIG =================
LINE_X = 320              # Vertical counting line (X position)
ZONE_WIDTH = 40           # Width of crossing zone (important for speed)
MIN_AREA = 1200           # Minimum contour area
 
# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
 
if not cap.isOpened():
    print("Error: Camera not opened")
    exit()
 
# ================= TRACKER =================
tracker = Sort(
    max_age=10,
    min_hits=2,
    iou_threshold=0.2
)
 
# ================= INIT FRAME =================
ret, prev = cap.read()
if not ret:
    print("Error: Cannot read first frame")
    exit()
 
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (5, 5), 0)
 
# ================= COUNTERS =================
people_in = 0
people_out = 0
states = {}   # track_id -> left / right / done
 
# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # ---------- FRAME DIFFERENCING ----------
    diff = cv2.absdiff(prev, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
 
    # ---------- MERGE BLOBS ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, None, iterations=2)
 
    # ---------- FIND CONTOURS ----------
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    detections = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
 
        x, y, w, h = cv2.boundingRect(c)
 
        # Reject very wide blobs (shadows)
        if w > h * 2:
            continue
 
        detections.append([x, y, x + w, y + h, 1.0])
 
    # SORT requires (N,5)
    detections = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(detections)
 
    # ---------- DRAW COUNTING ZONE ----------
    cv2.line(frame, (LINE_X - ZONE_WIDTH, 0),
             (LINE_X - ZONE_WIDTH, frame.shape[0]), (255, 255, 0), 1)
    cv2.line(frame, (LINE_X + ZONE_WIDTH, 0),
             (LINE_X + ZONE_WIDTH, frame.shape[0]), (255, 255, 0), 1)
 
    # ---------- PROCESS TRACKS ----------
    for t in tracks:
        x1, y1, x2, y2, tid = t.astype(int)
        cx = (x1 + x2) // 2
 
        # Determine zone
        if cx < LINE_X - ZONE_WIDTH:
            side = "left"
        elif cx > LINE_X + ZONE_WIDTH:
            side = "right"
        else:
            side = "center"
 
        # Initialize state
        if tid not in states:
            states[tid] = side
 
        # COUNTING (corrected direction)
        elif states[tid] == "left" and side == "right":
            people_out += 1
            states[tid] = "done"
 
        elif states[tid] == "right" and side == "left":
            people_in += 1
            states[tid] = "done"
 
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)
 
    # ---------- DISPLAY COUNTS ----------
    cv2.putText(frame, f"IN : {people_in}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {people_out}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    cv2.imshow("People Counter (Classical CV)", frame)
 
    prev = gray
 
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
 
# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
