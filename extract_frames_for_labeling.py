import os, csv, math, cv2

# ==== CONFIG ====
VIDEO_IN   = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/Game1.mp4"
EVENTS_CSV = "game_events/13Sep2025/game1_events.csv"   # columns like you posted
OUT_DIR    = "frames_output/13Sep2025"  # images go here
FPS_VIDEO  = 29.97

# Event-window sampling
WINDOW_BEFORE_S = 0.30
WINDOW_AFTER_S  = 0.50
FPS_WINDOW      = 10.0     # frames per second within event windows
MAX_FRAMES_PER_EVENT = 8   # hard cap per event window

# Periodic background sampling (outside event windows)
BASELINE_EVERY_N_FRAMES = 40   # one frame every N video frames

# Which events to window-sample
EVENTS_OF_INTEREST = {"throw","catch","tip","ground_in","ground_out","double"}

# ================
def safe(s):
    return (s or "").strip().replace(" ", "").replace("/", "_")

def load_events(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frame = int(float(row.get("frame","")))
            except:
                continue
            rows.append({
                "game_id":    row.get("game_id",""),
                "rally_id":   row.get("rally_id",""),
                "frame":      frame,
                "disc_color": row.get("disc_color",""),
                "event":      row.get("event",""),
                "event_team": row.get("event_team",""),
                "player":     row.get("player",""),
            })
    return rows

def time_to_frame(t, fps):
    return int(round(t * fps))

def frame_to_time(f, fps):
    return float(f) / float(fps)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    evs = load_events(EVENTS_CSV)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_IN}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_VIDEO

    # Build set of frames that belong to event windows (so we can avoid baseline duplicates)
    window_frames = set()
    per_event_samples = []

    for e in evs:
        evt = (e["event"] or "").lower()
        if evt not in EVENTS_OF_INTEREST:
            continue
        f0 = int(e["frame"])
        t0 = frame_to_time(f0, fps)

        t_start = max(0.0, t0 - WINDOW_BEFORE_S)
        t_end   = t0 + WINDOW_AFTER_S

        step = 1.0 / FPS_WINDOW
        t = t_start
        sampled = []
        while t <= t_end and len(sampled) < MAX_FRAMES_PER_EVENT:
            ff = time_to_frame(t, fps)
            if 0 <= ff < total_frames:
                window_frames.add(ff)
                sampled.append(ff)
            t += step

        per_event_samples.append({
            "event_row": e,
            "frames": sorted(sampled)
        })

    # Baseline sampling (outside event windows)
    baseline_frames = []
    for f in range(0, total_frames, BASELINE_EVERY_N_FRAMES):
        if f not in window_frames:
            baseline_frames.append(f)

    # Helper to grab & save a single frame
    def save_frame(frame_index, meta):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return False

        # filename: game_r{rally}_f{frame}_t{time}_{event}_{player}_{disc}.jpg
        t = frame_to_time(frame_index, fps)
        name = f"game{safe(meta.get('game_id','1'))}_r{safe(meta.get('rally_id',''))}" \
               f"_f{frame_index}_t{t:.2f}_{safe(meta.get('event','none'))}" \
               f"_{safe(meta.get('player',''))}_{safe(meta.get('disc_color',''))}.jpg"
        out_path = os.path.join(OUT_DIR, name)
        cv2.imwrite(out_path, frame)
        return True

    # Export event-window frames
    kept_ev = 0
    for block in per_event_samples:
        meta = block["event_row"]
        frames = block["frames"]
        # Choose a small subset to label: e.g., center (closest to event), one before, one after
        if not frames:
            continue
        # Always include the closest to event frame
        f0 = meta["frame"]
        frames_sorted = sorted(frames, key=lambda f: abs(f - f0))
        pick = []
        if frames_sorted:
            pick.append(frames_sorted[0])  # closest
        # try add one before and one after if available
        before = [f for f in frames_sorted if f < f0]
        after  = [f for f in frames_sorted if f > f0]
        if before:
            pick.append(before[0])
        if after:
            pick.append(after[0])

        # Save picks
        for ff in pick:
            if save_frame(ff, meta):
                kept_ev += 1

    # Export baseline negatives (use a neutral meta)
    kept_bg = 0
    for ff in baseline_frames:
        meta = {"game_id":"1","rally_id":"","event":"none","player":"","disc_color":""}
        if save_frame(ff, meta):
            kept_bg += 1

    cap.release()
    print(f"✅ Exported {kept_ev} event-centric frames and {kept_bg} baseline frames to {OUT_DIR}")
    print("Tip: In Roboflow, upload these, keep filenames, and use Batch -> Sort by filename to annotate in context.")
    
if __name__ == "__main__":
    main()
