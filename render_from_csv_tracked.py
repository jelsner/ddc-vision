# render_from_csv_tracked.py
import os
import csv
import cv2
import re
import numpy as np
from collections import defaultdict, deque

# === CONFIG ===
VIDEO_IN   = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/Game4.mov"
CSV_IN     = "Videos/Annotated/track_corrected_Game4.csv"   # frame-wise detections
VIDEO_OUT  = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/Game4a.mov"

# Homography (pixel -> meters) saved by your click tool (np.savetxt)
H_PATH     = "H.txt"

MIN_CONF         = 0.00  # keep everything already exported; raise if needed
TRAIL_LEN        = 48    # number of past centers to draw per track
TRAIL_STEP       = 1
BOX_THICK        = 3
SHOW_SPEED_FOR   = {"disc_red", "disc_yellow"}   # only show speed for these classes
USE_CORRECTED    = True  # prefer corrected_class when present
SKIP_EMPTY       = True  # skip rows whose (corrected_)class is blank
SKIP_BENCH       = True  # don't draw boxes/labels for player_bench rows
DRAW_PROXY_DASH  = True  # draw dashed outline for possession proxies (_is_proxy == 1)
PROXY_TAG        = True  # append "proxy" tag to label for proxies
# =============

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.4   # tuned for 4K; try 1.2–2.0
THICK      = 4
PAD_Y      = 12

# ---- 2D speed fallback (not used now, kept for reference) ----
SPEED_MAX_MPS = 50.0          # clamp scalar speed if needed

# --- 3D speed / height-proxy config (tune as needed) ---
ZPX_REF   = 40.0   # px: typical disc bbox size when it's ~Z_REF_M away (rough heuristic)
Z_REF_M   = 20.0   # m : distance corresponding to ZPX_REF (tune per lens/angle)
Z_MIN_M   = 0.0    # clamp
Z_MAX_M   = 12.0   # plausible apex height in meters (tune to your footage)

EMA_ALPHA_V   = 0.08   # EMA for speed readout (0..1; higher = snappier)
EMA_ALPHA_Z   = 0.20   # EMA for Z proxy (smoother height)
MAX_ACC_MPS2  = 80.0   # acceleration clamp for *velocity change* (try 50–120)

# BGR colors
RED     = (0,   0, 255)
YELLOW  = (0, 255, 255)
BLUE    = (255, 0,   0)
WHITE   = (255, 255, 255)
GREEN   = (0, 255, 0)
BLACK   = (0, 0, 0)
GRAY    = (180, 180, 180)

def class_to_color(name: str):
    n = (name or "").lower()
    # discs
    if "disc_red" in n:     return RED, RED
    if "disc_yellow" in n:  return YELLOW, YELLOW
    if n == "disc":         return GREEN, GREEN
    # players (team A / B by substring)
    if "playera" in n:      return BLUE, BLUE
    if "playerb" in n:      return WHITE, WHITE
    # generic player (if your CSV ever has 'player')
    if n == "player":       return GREEN, GREEN
    # bench
    if "player_bench" in n: return GRAY, GRAY
    # default
    return GREEN, GREEN

def to_xyxy(x, y, w, h):
    x1 = int(x - w/2.0); y1 = int(y - h/2.0)
    x2 = int(x + w/2.0); y2 = int(y + h/2.0)
    return x1, y1, x2, y2

def draw_label(img, text, org, color):
    x, y = org
    # outline
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, BLACK, THICK+2, lineType=cv2.LINE_AA)
    # main
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, THICK, lineType=cv2.LINE_AA)

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    """Simple dashed rectangle (used to show possession proxies)."""
    x1, y1 = pt1; x2, y2 = pt2
    def _dash_line(p1, p2):
        length = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        if length == 0: return
        vx = (p2[0]-p1[0]) / max(length, 1)
        vy = (p2[1]-p1[1]) / max(length, 1)
        pos = 0
        while pos < length:
            e = min(dash_len, length - pos)
            x_start = int(p1[0] + vx*pos); y_start = int(p1[1] + vy*pos)
            x_end   = int(p1[0] + vx*(pos+e)); y_end   = int(p1[1] + vy*(pos+e))
            cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)
            pos += dash_len + gap_len
    _dash_line((x1,y1), (x2,y1))
    _dash_line((x2,y1), (x2,y2))
    _dash_line((x2,y2), (x1,y2))
    _dash_line((x1,y2), (x1,y1))

def load_by_frame(csv_path):
    """
    Returns: dict[int_frame] -> list of detections with:
      track_id (int or -1), cls (string), conf (float), x,y,w,h (floats),
      time_s (float or None), is_proxy (bool)
    Prefers 'corrected_class' over 'class' if present; can skip blanks/bench.
    """
    by_frame = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        has_corr  = "corrected_class" in fields
        has_time  = "time_s" in fields
        has_proxy = "_is_proxy" in fields or "is_proxy" in fields

        for row in r:
            try:
                fidx = int(float(row.get("frame", 0)))
                conf = float(row.get("confidence") or 0.0)
                if conf < MIN_CONF:
                    continue

                # Prefer corrected class if present
                cls_raw = (row.get("corrected_class") if (USE_CORRECTED and has_corr) else row.get("class")) or ""
                cls = str(cls_raw).strip()

                if SKIP_EMPTY and (cls == "" or cls.lower() == "nan"):
                    continue
                if SKIP_BENCH and cls.lower() == "player_bench":
                    continue

                x = float(row["x"]); y = float(row["y"])
                w = float(row["width"]); h = float(row["height"])
            except Exception:
                continue

            # track id if present
            tid = -1
            if "track_id" in row and row["track_id"] not in (None, "", "nan"):
                try: tid = int(float(row["track_id"]))
                except: tid = -1

            t_s = None
            if has_time:
                try: t_s = float(row.get("time_s"))
                except: t_s = None

            is_proxy = False
            if has_proxy:
                pv = row.get("_is_proxy", row.get("is_proxy"))
                try:
                    is_proxy = (str(pv).strip() == "1")
                except:
                    is_proxy = False

            by_frame[fidx].append({
                "track_id": tid,
                "cls": cls,
                "conf": conf,
                "x": x, "y": y, "w": w, "h": h,
                "time_s": t_s,
                "is_proxy": is_proxy
            })
    return by_frame

def px_to_meters(cx_px, cy_px, H):
    """Apply homography (pixel->meters) to a center point."""
    v = np.array([cx_px, cy_px, 1.0], dtype=np.float64)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return None, None
    X = float(w[0] / w[2])
    Y = float(w[1] / w[2])
    return X, Y

def disc_height_proxy_m(box_w_px: float, box_h_px: float) -> float:
    """
    Quick-and-dirty Z proxy from apparent size: inverse proportional to the larger bbox edge.
    Z ≈ Z_REF_M * (ZPX_REF / max(major_px, 1)).
    Then clamped to [Z_MIN_M, Z_MAX_M].
    """
    bb = max(float(box_w_px), float(box_h_px), 1.0)
    z = Z_REF_M * (ZPX_REF / bb)
    return float(np.clip(z, Z_MIN_M, Z_MAX_M))

def clamp_velocity_change(v_raw: np.ndarray, v_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    Limit how fast velocity can change (|Δv| <= MAX_ACC_MPS2 * dt).
    If no prev velocity, return v_raw unchanged.
    """
    if v_prev is None:
        return v_raw
    dv = v_raw - v_prev
    dv_mag = float(np.linalg.norm(dv))
    if dv_mag <= 1e-9:
        return v_raw
    dv_max = MAX_ACC_MPS2 * max(dt, 1e-3)
    if dv_mag <= dv_max:
        return v_raw
    return v_prev + dv * (dv_max / dv_mag)

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    assert os.path.exists(CSV_IN),   f"CSV not found: {CSV_IN}"
    assert os.path.exists(H_PATH),   f"Homography not found: {H_PATH}"

    Hm = np.loadtxt(H_PATH)
    preds = load_by_frame(CSV_IN)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open input video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_px))

    # trail state: track_id -> deque[(x,y), ...]
    trails = {}
    # kinematic state: tid -> {last_XYZ_m, last_t, ema_v, v_prev, ema_z}
    kin = {}

    f = 0
    total_boxes = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = preds.get(f, [])
        default_t = f / FPS  # used if no time_s

        # fast lookup for last class per tid (for trail color)
        last_cls_for_tid = {}
        for d in dets:
            if int(d.get("track_id", -1)) != -1:
                last_cls_for_tid[int(d["track_id"])] = d["cls"]

        for d in dets:
            cls_lc = (d["cls"] or "").lower()

            if SKIP_BENCH and cls_lc == "player_bench":
                continue
            if SKIP_EMPTY and (cls_lc == "" or cls_lc == "nan"):
                continue

            x1, y1, x2, y2 = to_xyxy(d["x"], d["y"], d["w"], d["h"])
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            t_now = d["time_s"] if d["time_s"] is not None else default_t

            # metric center via homography
            X_m, Y_m = px_to_meters(cx, cy, Hm)

            # 3D speed with Z-proxy (disc only), EMA + acceleration clamp
            spd = None
            tid = int(d.get("track_id", -1))
            is_disc = ("disc" in cls_lc)

            if tid != -1 and X_m is not None and Y_m is not None:
                z_proxy = disc_height_proxy_m(d["w"], d["h"]) if is_disc else 0.0
                prev = kin.get(tid)

                if prev is None:
                    kin[tid] = {
                        "last_XYZ_m": np.array([X_m, Y_m, z_proxy], dtype=np.float64),
                        "last_t": float(t_now),
                        "ema_v": 0.0,
                        "v_prev": None,
                        "ema_z": float(z_proxy),
                    }
                    spd = 0.0
                else:
                    prev["ema_z"] = float(EMA_ALPHA_Z * z_proxy + (1.0 - EMA_ALPHA_Z) * prev["ema_z"])
                    Z_m = prev["ema_z"] if is_disc else 0.0

                    p_now = np.array([X_m, Y_m, Z_m], dtype=np.float64)
                    dt = max(1e-3, float(t_now - prev["last_t"]))  # seconds

                    v_raw = (p_now - prev["last_XYZ_m"]) / dt
                    v_clamped = clamp_velocity_change(v_raw, prev.get("v_prev"), dt)

                    v_mag = float(np.linalg.norm(v_clamped))
                    ema_v = EMA_ALPHA_V * v_mag + (1.0 - EMA_ALPHA_V) * prev["ema_v"]
                    ema_v = float(np.clip(ema_v, 0.0, SPEED_MAX_MPS))

                    prev["last_XYZ_m"] = p_now
                    prev["last_t"]     = float(t_now)
                    prev["v_prev"]     = v_clamped
                    prev["ema_v"]      = ema_v

                    spd = ema_v

            # draw box + label
            box_color, text_color = class_to_color(d["cls"])
            if DRAW_PROXY_DASH and d.get("is_proxy", False):
                draw_dashed_rect(frame, (x1, y1), (x2, y2), box_color, thickness=max(1, BOX_THICK-1))
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICK)

            lbl_base = d["cls"]  # already corrected when present
            if tid != -1:
                lbl_base += f" #{tid}"

            # only show speed for discs (quieter HUD)
            add_speed = (lbl_base.split()[0].lower() in SHOW_SPEED_FOR) and (spd is not None)

            extra = ""
            if PROXY_TAG and d.get("is_proxy", False):
                extra = " proxy"

            if add_speed:
                label = f"{lbl_base} {d['conf']:.2f}  {spd:.1f} m/s{extra}"
            else:
                label = f"{lbl_base} {d['conf']:.2f}{extra}"

            draw_label(frame, label, (x1, max(0, y1 - PAD_Y)), text_color)

            # trails (by tid)
            if tid != -1:
                if tid not in trails:
                    trails[tid] = deque(maxlen=TRAIL_LEN)
                trails[tid].append((cx, cy))
            total_boxes += 1

        # draw trails (under labels so they don’t obscure text)
        # choose trail color by last known class this frame; fallback gray
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            cls_for_tid = None
            # pick from current frame dets if available
            for d in dets:
                if int(d.get("track_id", -1)) == tid:
                    cls_for_tid = d["cls"]
                    break
            trail_color = class_to_color(cls_for_tid)[0] if cls_for_tid else GRAY

            for i in range(len(pts) - 1, 0, -TRAIL_STEP):
                p1 = pts[i]; p2 = pts[i - 1]
                t = int(np.interp(i, [1, len(pts)-1], [1, 5]))  # fade thickness
                cv2.line(frame, p1, p2, trail_color, t)

        # HUD
        draw_label(frame, f"frame={f} det={len(dets)}", (12, 50), WHITE)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"✅ Saved: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
