# ddc_fsm_from_sheet.py
import os
import math
import pandas as pd

# ---------- CONFIG ----------
INPUT_CSV         = "game_events/13Sep2025/game2_events.csv"          # your sheet export as CSV
OUT_SUMMARY_CSV   = "rally_summaries/13Sep2025/game2_summary.csv"
OUT_WARNINGS_CSV  = "rally_summaries/13Sep2025/game2_warnings.csv"
FPS               = 29.97                       # edited-video frame rate
# ----------------------------

# ---- Normalizers ------------------------------------------------------------

def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def norm_color(x):
    s = norm_str(x).lower()
    if s in ("red", "r"):
        return "red"
    if s in ("yellow", "y"):
        return "yellow"
    # allow rally-level, non-disc events (e.g., double, switch_sides)
    return ""  # empty means “not a disc-specific row”

def norm_team(x):
    s = norm_str(x).upper()
    if s in ("A", "B"):
        return s
    return ""  # unknown/empty

def norm_player(x):
    return norm_str(x)

def norm_event(x):
    s = norm_str(x).lower()
    # unify the throw-like events
    if s in ("serve", "lead", "attack", "initiate", "throw"):
        return "throw"
    # common events we care about
    if s in ("catch", "tip", "double", "ground_in", "ground_out", "switch_sides"):
        return s
    return s  # leave as-is (won’t break anything; may trigger a warning)

def parse_score(x):
    if pd.isna(x) or str(x).strip() == "":
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0

# ---- FSM (simple rally-level pass) -----------------------------------------

def build_fsm(df):
    """
    Minimal per-rally aggregation with sanity checks.
    Assumes columns:
      game_id, rally_id, frame, disc_color, event, event_team, score_team, player, score, details
    """
    warns = []
    # normalize fields
    df = df.copy()

    # Ensure columns exist (create blank if missing)
    expected = ["game_id","rally_id","frame","disc_color","event","event_team","score_team","player","score","details"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    # normalize
    df["game_id"]     = df["game_id"]    .apply(norm_str)
    df["rally_id"]    = df["rally_id"]   .apply(lambda x: int(float(x)) if str(x).strip() != "" and not pd.isna(x) else -1)
    df["frame"]       = df["frame"]      .apply(lambda x: int(float(x)) if str(x).strip() != "" and not pd.isna(x) else -1)
    df["disc_color_n"]= df["disc_color"] .apply(norm_color)
    df["event_n"]     = df["event"]      .apply(norm_event)
    df["event_team_n"]= df["event_team"] .apply(norm_team)
    df["score_team_n"]= df["score_team"] .apply(norm_team)
    df["player_n"]    = df["player"]     .apply(norm_player)
    df["score_n"]     = df["score"]      .apply(parse_score)
    df["details_n"]   = df["details"]    .apply(norm_str)

    # drop rows without a rally_id or frame
    df = df[df["rally_id"] >= 0]
    df = df[df["frame"] >= 0]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame([{"warning":"No usable rows after normalization."}])

    # sort
    df.sort_values(["game_id","rally_id","frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    summary_rows = []
    warns_rows = []

    # group by rally
    for (game_id, rally_id), g in df.groupby(["game_id","rally_id"], sort=True):
        g = g.sort_values("frame").copy()

        start_frame = int(g["frame"].min())
        end_frame   = int(g["frame"].max())
        duration_s  = (end_frame - start_frame) / FPS if FPS > 0 else math.nan

        # serves (first throws)
        throws = g[g["event_n"]=="throw"]
        serve_A = None
        serve_B = None
        if not throws.empty:
            first_A = throws[throws["event_team_n"]=="A"]
            first_B = throws[throws["event_team_n"]=="B"]
            if not first_A.empty:
                serve_A = int(first_A["frame"].iloc[0])
            if not first_B.empty:
                serve_B = int(first_B["frame"].iloc[0])

        # scoring: trust explicit score rows if present
        # (e.g., double with score=2, ground_in/out with score=1)
        score_rows = g[g["score_n"] != 0]
        pts_A = int(score_rows[score_rows["score_team_n"]=="A"]["score_n"].sum())
        pts_B = int(score_rows[score_rows["score_team_n"]=="B"]["score_n"].sum())

        # double present?
        double_present = any((g["event_n"]=="double") & (g["score_n"]==2))
        double_team = ""
        if double_present:
            # if multiple doubles, take the last one’s score_team
            last_double = g[(g["event_n"]=="double") & (g["score_n"]==2)].iloc[-1]
            double_team = last_double.get("score_team_n","") or ""

        # sanity warnings
        # 1) No throws at the start
        if throws.empty:
            warns_rows.append({
                "game_id": game_id,
                "rally_id": rally_id,
                "frame": start_frame,
                "warning": "Rally has no 'throw' events."
            })
        # 2) Has score without any catch/ground events
        has_score = (pts_A + pts_B) > 0
        has_ball_end = any(g["event_n"].isin(["catch","ground_in","ground_out","double"]))
        if has_score and not has_ball_end:
            warns_rows.append({
                "game_id": game_id,
                "rally_id": rally_id,
                "frame": end_frame,
                "warning": "Points recorded but no terminal event (catch/ground/double) found."
            })
        # 3) Double present but recorded points don’t sum to 2 for exactly one team
        if double_present:
            if (pts_A == 2 and pts_B == 0) or (pts_B == 2 and pts_A == 0):
                pass
            else:
                warns_rows.append({
                    "game_id": game_id,
                    "rally_id": rally_id,
                    "frame": end_frame,
                    "warning": f"'double' present but point totals are A={pts_A}, B={pts_B}."
                })

        # per-disc consistency checks (red/yellow never “switch color” mid-rally)
        for disc in ["red","yellow"]:
            dg = g[g["disc_color_n"]==disc]
            # ensure disc rows are in non-decreasing frame order (they should be)
            if not dg.empty:
                frames = dg["frame"].values
                if any(frames[i] > frames[i+1] for i in range(len(frames)-1)):
                    warns_rows.append({
                        "game_id": game_id,
                        "rally_id": rally_id,
                        "frame": int(dg["frame"].min()),
                        "warning": f"{disc} rows not monotonic in frame order."
                    })

        summary_rows.append({
            "game_id": game_id,
            "rally_id": rally_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_s": round(duration_s, 3),
            "serve_A_frame": serve_A if serve_A is not None else "",
            "serve_B_frame": serve_B if serve_B is not None else "",
            "double_present": bool(double_present),
            "double_team": double_team,
            "points_A": pts_A,
            "points_B": pts_B
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["game_id","rally_id"])
    warns_df   = pd.DataFrame(warns_rows).sort_values(["game_id","rally_id","frame"]) if warns_rows else pd.DataFrame(columns=["game_id","rally_id","frame","warning"])
    return summary_df, warns_df

# ---- Main -------------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    summary_df, warns_df = build_fsm(df)

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    warns_df.to_csv(OUT_WARNINGS_CSV, index=False)

    print(f"✅ Wrote {OUT_SUMMARY_CSV} ({len(summary_df)} rallies)")
    print(f"⚠️  Wrote {OUT_WARNINGS_CSV} ({len(warns_df)} warnings)")

if __name__ == "__main__":
    main()
