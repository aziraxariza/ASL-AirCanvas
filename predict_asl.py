import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import threading
import queue
import subprocess

# ===============================
# Load Model
# ===============================
model  = load_model("asl_model.h5")
labels = np.load("label_classes.npy", allow_pickle=True)

# ===============================
# Speech Engine
# ===============================
# pyttsx3 on Windows: COM event loop dies after first runAndWait().
# Fix: drive SAPI directly via PowerShell subprocess — always fresh COM,
# runs in daemon thread so CV loop is never blocked.

_speech_queue = queue.Queue()
_current_proc = None

def _speech_worker():
    global _current_proc
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        if _current_proc and _current_proc.poll() is None:
            _current_proc.terminate()
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 6; "
            f'$s.Speak("{text}");'
        )
        _current_proc = subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _current_proc.wait()
        _speech_queue.task_done()

threading.Thread(target=_speech_worker, daemon=True).start()

last_speak_time = 0.0

def _clear_queue():
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
            _speech_queue.task_done()
        except Exception:
            pass

def speak(text, cooldown=0.5):
    global last_speak_time
    if not text.strip():
        return
    now = time.time()
    if now - last_speak_time < cooldown:
        return
    last_speak_time = now
    _clear_queue()
    _speech_queue.put(text)

def speak_force(text):
    global last_speak_time
    if not text.strip():
        return
    last_speak_time = time.time()
    _clear_queue()
    _speech_queue.put(text)

# ===============================
# MediaPipe
# ===============================
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1)
mp_draw  = mp.solutions.drawing_utils

LM_STYLE   = mp_draw.DrawingSpec(color=(60, 220, 255), thickness=1, circle_radius=2)
CONN_STYLE = mp_draw.DrawingSpec(color=(30, 100, 160), thickness=1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

CONFIDENCE_THRESHOLD = 0.90
STABILITY_FRAMES     = 4
PREDICT_EVERY_N      = 2    # run inference every 2nd frame only

word             = ""
last_prediction  = ""
stable_count     = 0
frame_count      = 0
cached_label     = ""   # holds last valid prediction across skipped frames
cached_conf      = 0.0
last_added_label = ""

canvas         = None
prev_x, prev_y = None, None
drawing_mode   = False

DRAW_COLOR = (60, 220, 255)
DRAW_THICK = 3

flash_frames = 0
flash_letter = ""

# ===============================
# UI Helpers
# ===============================

def glass_rect(img, x1, y1, x2, y2, color, alpha=0.55, r=10):
    ov = img.copy()
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def border_rect(img, x1, y1, x2, y2, color, t=1, r=10):
    cv2.rectangle(img, (x1+r, y1),   (x2-r, y1),   color, t)
    cv2.rectangle(img, (x1+r, y2),   (x2-r, y2),   color, t)
    cv2.rectangle(img, (x1, y1+r),   (x1, y2-r),   color, t)
    cv2.rectangle(img, (x2, y1+r),   (x2, y2-r),   color, t)
    for (cx, cy, a1, a2) in [
        (x1+r, y1+r, 180, 270), (x2-r, y1+r, 270, 360),
        (x1+r, y2-r,  90, 180), (x2-r, y2-r,   0,  90)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, a1, a2, color, t)

def gradient_bar(img, x1, y1, x2, y2, ratio, c0, c1, r=5):
    glass_rect(img, x1, y1, x2, y2, (8, 8, 18), alpha=0.80, r=r)
    fw = int(ratio * (x2-x1-4))
    for i in range(fw):
        t = i / max(fw-1, 1)
        col = tuple(int(c0[c] + t*(c1[c]-c0[c])) for c in range(3))
        cv2.line(img, (x1+2+i, y1+2), (x1+2+i, y2-2), col, 1)

def outlined_text(img, text, pos, font, scale, color, t=1, oc=(0,0,0), ot=3):
    cv2.putText(img, text, pos, font, scale, oc, ot, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, t, cv2.LINE_AA)

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_label = cached_label   # default to last known — no flicker on skipped frames
    confidence      = cached_conf

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                               LM_STYLE, CONN_STYLE)

        tip = handLms.landmark[8]
        x   = int(tip.x * W)
        y   = int(tip.y * H)

        if drawing_mode:
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), DRAW_COLOR, DRAW_THICK)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None

        lm_list = []
        for lm in handLms.landmark:
            lm_list += [lm.x, lm.y]

        # Skip prediction while drawing; also skip every other frame to cut latency
        frame_count += 1
        if not drawing_mode and frame_count % PREDICT_EVERY_N == 0:
            pred            = model.predict(np.array([lm_list]), verbose=0)
            confidence      = float(np.max(pred))
            predicted_label = labels[np.argmax(pred)]

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_label = ""

            # Update cache so skipped frames show the last real result
            cached_label = predicted_label
            cached_conf  = confidence

            if predicted_label == last_prediction and predicted_label != "":
                stable_count += 1
            else:
                stable_count = 0
                last_added_label = ""

            if stable_count >= STABILITY_FRAMES and predicted_label != last_added_label:
                if len(predicted_label) == 1:
                    word += predicted_label
                else:
                    pl = predicted_label.lower()
                    if pl == "space":
                        word += " "
                    elif pl == "del":
                        word = word[:-1]
                    else:
                        word = predicted_label

                last_added_label = predicted_label
                flash_letter      = predicted_label
                flash_frames      = 14
                speak(predicted_label, cooldown=0.3)
                stable_count = 0

            last_prediction = predicted_label

    else:
        stable_count    = 0
        last_prediction = ""
        prev_x, prev_y  = None, None
        cached_label    = ""   # hand gone — clear cache
        cached_conf     = 0.0

    if flash_frames > 0:
        flash_frames -= 1

    # Merge canvas
    gray_c  = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, msk   = cv2.threshold(gray_c, 1, 255, cv2.THRESH_BINARY)
    msk_inv  = cv2.bitwise_not(msk)
    frame    = cv2.add(cv2.bitwise_and(frame,  frame,  mask=msk_inv),
                       cv2.bitwise_and(canvas, canvas, mask=msk))

    # =========================================================================
    # HUD — glassmorphic dark sci-fi translator overlay
    # =========================================================================
    CYAN   = (255, 225,  45)   # BGR  warm cyan
    PURPLE = (215,  65, 155)   #       violet accent
    GREEN  = ( 55, 215,  95)   #       acceptance green
    ORANGE = ( 25, 140, 255)   #       warning orange
    WHITE  = (255, 255, 255)
    DIM    = (130, 130, 145)
    PANEL  = (  8,   8,  18)
    GLOW   = (200, 245, 255)   # flash white-cyan

    # ── top-left detection panel (220 x 130) ─────────────────────────────────
    PX, PY, PW, PH = 12, 12, 220, 130
    glass_rect(frame, PX, PY, PX+PW, PY+PH, PANEL, alpha=0.70, r=12)
    bc = GLOW if flash_frames > 0 else CYAN
    border_rect(frame, PX, PY, PX+PW, PY+PH, bc, t=1, r=12)

    cv2.putText(frame, "DETECTED", (PX+12, PY+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, CYAN, 1, cv2.LINE_AA)

    disp = predicted_label if predicted_label else "\u2014"
    lc   = GLOW if flash_frames > 0 else CYAN
    cv2.putText(frame, disp, (PX+14, PY+108),
                cv2.FONT_HERSHEY_DUPLEX, 3.1, (15, 50, 55), 12, cv2.LINE_AA)
    cv2.putText(frame, disp, (PX+14, PY+108),
                cv2.FONT_HERSHEY_DUPLEX, 3.1, lc, 2, cv2.LINE_AA)

    conf_pct = int(confidence * 100)
    pc       = GREEN if confidence >= 0.90 else (ORANGE if confidence >= 0.70 else DIM)
    glass_rect(frame, PX+108, PY+70, PX+PW-4, PY+96, (14,14,28), alpha=0.88, r=7)
    cv2.putText(frame, f"{conf_pct}%  conf", (PX+116, PY+89),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, pc, 1, cv2.LINE_AA)

    # ── stability bar ─────────────────────────────────────────────────────────
    BX1, BY1 = PX, PY+PH+6
    BX2, BY2 = PX+PW, BY1+14
    gradient_bar(frame, BX1, BY1, BX2, BY2,
                 stable_count / STABILITY_FRAMES,
                 (18, 70, 70), CYAN, r=5)
    cv2.putText(frame, "STABILITY", (BX2+8, BY2-1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.31, DIM, 1, cv2.LINE_AA)

    # ── top-right mode badge ──────────────────────────────────────────────────
    BW = 156
    MX1, MX2 = W-BW-12, W-12
    if drawing_mode:
        glass_rect(frame, MX1, 12, MX2, 46, (0, 38, 38), alpha=0.78, r=10)
        border_rect(frame, MX1, 12, MX2, 46, CYAN, t=1, r=10)
        cv2.putText(frame, "DRAW  ON",  (MX1+14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1, cv2.LINE_AA)
    else:
        glass_rect(frame, MX1, 12, MX2, 46, PANEL, alpha=0.65, r=10)
        border_rect(frame, MX1, 12, MX2, 46, (55,55,65), t=1, r=10)
        cv2.putText(frame, "DRAW  OFF", (MX1+14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, DIM, 1, cv2.LINE_AA)

    # ── bottom word panel ─────────────────────────────────────────────────────
    BOTT_H = 58
    glass_rect(frame, 0, H-BOTT_H, W, H, PANEL, alpha=0.82, r=0)
    cv2.line(frame, (0, H-BOTT_H), (W, H-BOTT_H), PURPLE, 1)

    cv2.putText(frame, "WORD", (16, H-BOTT_H+17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, PURPLE, 1, cv2.LINE_AA)

    dword = (word[-32:] if len(word) > 32 else word) or "\u2014"
    outlined_text(frame, dword, (16, H-12),
                  cv2.FONT_HERSHEY_DUPLEX, 0.95, WHITE,
                  t=1, oc=(28, 8, 38), ot=4)

    # hotkeys right-aligned in bottom panel
    hotkeys = [("D","Draw"),("X","Clear"),("SPC","Speak"),("C","Reset"),("ESC","Quit")]
    kx = W - 12
    for k, lbl in reversed(hotkeys):
        full = f"{k}:{lbl}"
        (tw, _), _ = cv2.getTextSize(full, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        kx -= tw + 16
        (kw, _), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(frame, k,         (kx,    H-BOTT_H+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ORANGE, 1, cv2.LINE_AA)
        cv2.putText(frame, f":{lbl}", (kx+kw, H-BOTT_H+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM,    1, cv2.LINE_AA)

    # ── fingertip crosshair cursor (drawing mode) ─────────────────────────────
    if drawing_mode and result.multi_hand_landmarks:
        tip = result.multi_hand_landmarks[0].landmark[8]
        cx_ = int(tip.x * W)
        cy_ = int(tip.y * H)
        cv2.circle(frame, (cx_, cy_), 13, CYAN,  1, cv2.LINE_AA)
        cv2.circle(frame, (cx_, cy_),  4, WHITE, -1, cv2.LINE_AA)
        for dx, dy in [(20,0),(-20,0),(0,20),(0,-20)]:
            cv2.line(frame, (cx_+dx//3, cy_+dy//3),
                             (cx_+dx,   cy_+dy), CYAN, 1, cv2.LINE_AA)

    cv2.imshow("ASL  \u2736  Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        drawing_mode     = not drawing_mode
        prev_x, prev_y   = None, None
        # reset prediction state so nothing carries over from draw mode
        stable_count     = 0
        last_prediction  = ""
        last_added_label = ""
        predicted_label  = ""
        confidence       = 0.0
        cached_label     = ""
        cached_conf      = 0.0

    if key == ord('x'):
        canvas = np.zeros_like(frame)

    if key == 32:
        speak_force(word if word else "nothing")

    if key == ord('c'):
        word             = ""
        last_added_label = ""

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()