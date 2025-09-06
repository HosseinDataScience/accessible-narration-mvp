# build_accessible_narration_en.py
# Robust MVP narration builder (English) – resilient to missing columns/files
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Paths & IO -----------------------------
ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")

# Core inputs
TL_CSV      = ROOT / r"mvp_outputs\merged\unified_timeline.csv"       # dialogue + sfx (merged)
SCENE_V_CSV = ROOT / r"mvp_outputs\vision\scene_vision.csv"           # optional scenes
SCENE_V_JL  = ROOT / r"mvp_outputs\vision\scene_vision.jsonl"         # optional details per keyframe
PANN_EVT    = ROOT / r"mvp_outputs\events\panns_other_events.csv"     # optional sfx events
POSE_CSV    = ROOT / r"mvp_outputs\pose_emotion\pose_per_frame.csv"   # optional pose per frame
EMO_CSV     = ROOT / r"mvp_outputs\pose_emotion\emotion_per_face.csv" # optional face emotions
FACE_SCN    = ROOT / r"mvp_outputs\faces\scene_characters.csv"        # optional per-scene characters

# Outputs
OUT_DIR     = ROOT / r"mvp_outputs\narration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT     = OUT_DIR / "accessible_narration_en.txt"
OUT_CSV     = OUT_DIR / "accessible_narration_en.csv"


# ----------------------------- Helpers -----------------------------
def sec(t: float) -> str:
    """Format seconds to mm:ss.s"""
    if pd.isna(t):
        return "?"
    m, s = divmod(float(t), 60.0)
    return f"{int(m):02d}:{s:04.1f}"

def read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def read_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items

def ensure_intervals(df: pd.DataFrame, time_col: str = "time_s", default_dt: float = 0.033) -> pd.DataFrame:
    """
    If df lacks start_s/end_s, but has time_s (or similar), create small intervals.
    """
    if df is None or df.empty:
        return df
    cols_lower = {c.lower() for c in df.columns}
    if "start_s" in cols_lower and "end_s" in cols_lower:
        return df

    # try find time column
    cand = [c for c in df.columns if c.lower() == time_col.lower()]
    if not cand:
        cand = [c for c in df.columns if c.lower() in ["time", "t", "ts", "timestamp", "sec", "seconds"]]
    if not cand:
        return df

    tcol = cand[0]
    t = pd.to_numeric(df[tcol], errors="coerce").values
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return df

    dt = default_dt
    if len(t) >= 2:
        diffs = np.diff(np.sort(t))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            dt = float(np.median(diffs))

    df = df.copy()
    df["start_s"] = pd.to_numeric(df[tcol], errors="coerce")
    df["end_s"]   = df["start_s"] + dt
    return df

def overlap_mask(df: pd.DataFrame, s0: float, s1: float) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=bool)
    cols = [c.lower() for c in df.columns]
    if "start_s" not in cols or "end_s" not in cols:
        return pd.Series([False] * len(df))
    # make sure numeric
    dfx = df.copy()
    dfx["start_s"] = pd.to_numeric(dfx[df.columns[cols.index("start_s")]], errors="coerce")
    dfx["end_s"]   = pd.to_numeric(dfx[df.columns[cols.index("end_s")]], errors="coerce")
    return (dfx["start_s"] < s1) & (dfx["end_s"] > s0)

def short_list(values: List[str], k: int = 4) -> str:
    values = [v for v in values if isinstance(v, str) and v.strip()]
    if not values:
        return ""
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
        if len(uniq) >= k:
            break
    return ", ".join(uniq)


# ----------------------------- Load Inputs -----------------------------
df_tl    = read_csv_safe(TL_CSV)
df_pose  = ensure_intervals(read_csv_safe(POSE_CSV), time_col="time_s", default_dt=0.033)
df_emo   = ensure_intervals(read_csv_safe(EMO_CSV),  time_col="time_s", default_dt=0.033)
df_panns = ensure_intervals(read_csv_safe(PANN_EVT), time_col="time_s", default_dt=0.1)  # events usually have intervals
df_faces = ensure_intervals(read_csv_safe(FACE_SCN), time_col="time_s", default_dt=0.1)

scn_csv  = read_csv_safe(SCENE_V_CSV)
scn_jl   = read_jsonl_safe(SCENE_V_JL)  # list of dicts


# ----------------------------- Build Scene List -----------------------------
scenes: List[Dict[str, Any]] = []
if not scn_csv.empty and {"start_s", "end_s"}.issubset({c.lower() for c in scn_csv.columns}):
    # normalize column names
    cc = {c.lower(): c for c in scn_csv.columns}
    for _, r in scn_csv.iterrows():
        s0 = float(r[cc["start_s"]])
        s1 = float(r[cc["end_s"]])
        sid = int(r[cc["scene_idx"]]) if "scene_idx" in cc else len(scenes)
        scenes.append({"scene_idx": sid, "start_s": s0, "end_s": s1})
else:
    # Fallback: single scene covering whole timeline
    if not df_tl.empty:
        s0 = float(pd.to_numeric(df_tl.get("start_s", pd.Series([0])), errors="coerce").min())
        s1 = float(pd.to_numeric(df_tl.get("end_s",   pd.Series([0])), errors="coerce").max())
        if not (np.isfinite(s0) and np.isfinite(s1)) or s1 <= s0:
            s0 = 0.0
            # try duration from tl
            s1 = float(pd.to_numeric(df_tl.get("start_s", pd.Series([0])), errors="coerce").max()) + 1.0
        scenes.append({"scene_idx": 0, "start_s": s0, "end_s": s1})


# ----------------------------- Vision Index (optional) -----------------------------
# Build quick access from JSONL by time if available
vision_by_time: List[Dict[str, Any]] = scn_jl  # expect keys like: t_s, blip_caption, objects(list)
# If missing, keep empty and skip in enrich


# ----------------------------- Enrichment -----------------------------
def enrich_scene_text(scene: Dict[str, Any]) -> str:
    s0, s1 = float(scene["start_s"]), float(scene["end_s"])
    parts: List[str] = []

    # Vision (captions + objects) from JSONL
    if vision_by_time:
        # take items within scene window
        vis = []
        objs = []
        for it in vision_by_time:
            t_s = float(it.get("t_s", it.get("time_s", -1)))
            if not np.isfinite(t_s) or not (s0 <= t_s <= s1):
                continue
            cap = it.get("blip_caption") or it.get("caption") or ""
            if isinstance(cap, str) and cap.strip():
                vis.append(cap.strip())
            olist = it.get("objects") or it.get("yolo_objects") or []
            if isinstance(olist, list):
                for o in olist:
                    if isinstance(o, str):
                        objs.append(o)
                    elif isinstance(o, dict) and "label" in o:
                        objs.append(str(o["label"]))
        cap_txt = ""
        if vis:
            # pick up to 2 diverse captions
            cap_txt = short_list(vis, k=2)
        obj_txt = short_list(objs, k=6)
        if cap_txt:
            parts.append(f"visual focus: {cap_txt}")
        if obj_txt:
            parts.append(f"key objects: {obj_txt}")

    # Faces / characters (optional)
    if not df_faces.empty and {"character", "start_s", "end_s"}.issubset({c.lower() for c in df_faces.columns}):
        cc = {c.lower(): c for c in df_faces.columns}
        m = overlap_mask(df_faces, s0, s1)
        chars = df_faces.loc[m, cc.get("character", "character")].astype(str).tolist()
        ch_txt = short_list(chars, k=5)
        if ch_txt:
            parts.append(f"present characters: {ch_txt}")

    # Pose (optional)
    if not df_pose.empty and {"start_s", "end_s"}.issubset({c.lower() for c in df_pose.columns}):
        cc = {c.lower(): c for c in df_pose.columns}
        m = overlap_mask(df_pose, s0, s1)
        if m.any():
            pose_col = None
            for k in ["pose_label", "pose", "body_pose", "activity"]:
                if k in cc:
                    pose_col = cc[k]
                    break
            if pose_col:
                poses = df_pose.loc[m, pose_col].astype(str).tolist()
                pose_txt = short_list(poses, k=4)
                if pose_txt:
                    parts.append(f"body poses: {pose_txt}")

    # Emotion (optional)
    if not df_emo.empty and {"start_s", "end_s"}.issubset({c.lower() for c in df_emo.columns}):
        cc = {c.lower(): c for c in df_emo.columns}
        m = overlap_mask(df_emo, s0, s1)
        if m.any():
            emo_col = None
            for k in ["dominant_emotion", "emotion", "face_emotion"]:
                if k in cc:
                    emo_col = cc[k]
                    break
            if emo_col:
                emos = df_emo.loc[m, emo_col].astype(str).tolist()
                emo_txt = short_list(emos, k=4)
                if emo_txt:
                    parts.append(f"likely facial emotions: {emo_txt}")

    # PANNs events (optional soundscape cues)
    if not df_panns.empty and {"start_s", "end_s", "label"}.issubset({c.lower() for c in df_panns.columns}):
        cc = {c.lower(): c for c in df_panns.columns}
        m = overlap_mask(df_panns, s0, s1)
        if m.any():
            ev = df_panns.loc[m, cc["label"]].astype(str).tolist()
            ev_txt = short_list(ev, k=5)
            if ev_txt:
                parts.append(f"ambient sounds: {ev_txt}")

    return " | ".join(parts) if parts else " "


# ----------------------------- Narration Build -----------------------------
def build_scene_narration(scene: Dict[str, Any], df_tl: pd.DataFrame) -> Tuple[str, List[Dict[str, Any]]]:
    s0, s1 = float(scene["start_s"]), float(scene["end_s"])
    out_lines: List[str] = []
    out_rows: List[Dict[str, Any]] = []

    intro = f"Scene {scene['scene_idx']} ({sec(s0)}–{sec(s1)}): {enrich_scene_text(scene)}"
    out_lines.append(intro)
    out_rows.append({
        "scene_idx": scene["scene_idx"],
        "kind": "scene_intro",
        "start_s": s0,
        "end_s": s1,
        "text": intro
    })

    if df_tl.empty:
        return "\n".join(out_lines), out_rows

    # filter timeline rows overlapping this scene
    tl = df_tl.copy()
    for col in ["start_s", "end_s"]:
        tl[col] = pd.to_numeric(tl.get(col), errors="coerce")
    m = (tl["start_s"] < s1) & (tl["end_s"] > s0)
    tl = tl.loc[m].sort_values(["start_s", "end_s"], na_position="last")

    # choose columns
    kind_col = "kind" if "kind" in tl.columns else None
    text_col = "content" if "content" in tl.columns else None
    extra_col = "extra" if "extra" in tl.columns else None

    for _, r in tl.iterrows():
        k  = str(r.get(kind_col, "") or "").strip().lower() if kind_col else ""
        ts = float(r.get("start_s", s0))
        te = float(r.get("end_s",   ts))
        txt = str(r.get(text_col, "") or "").strip() if text_col else ""
        extra = str(r.get(extra_col, "") or "").strip() if extra_col else ""

        if k == "dialogue":
            line = f"[{sec(ts)}] Dialogue: {txt}"
        elif k == "sfx":
            # try parse label=XYZ;score=0.42 pattern if present
            label = ""
            if "label=" in extra:
                try:
                    # crude parse
                    chunk = extra.split("label=")[1]
                    label = chunk.split(";")[0].strip()
                except Exception:
                    pass
            line = f"[{sec(ts)}] SFX: {label or txt or extra}"
        else:
            # fallback
            base = txt or extra or "(event)"
            line = f"[{sec(ts)}] {k or 'event'}: {base}"

        out_lines.append(line)
        out_rows.append({
            "scene_idx": scene["scene_idx"],
            "kind": k or "event",
            "start_s": ts,
            "end_s": te,
            "text": line
        })

    out_lines.append("")  # blank line between scenes
    return "\n".join(out_lines), out_rows


# ----------------------------- Run -----------------------------
all_lines: List[str] = []
all_rows: List[Dict[str, Any]] = []

if not scenes:
    all_lines.append("No scenes detected. Nothing to narrate.")
else:
    for sc in scenes:
        scene_text, rows = build_scene_narration(sc, df_tl)
        all_lines.append(scene_text)
        all_rows.extend(rows)

# Save TXT
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines).strip() + "\n")

# Save CSV
pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

print("Saved narration:")
print(" -", OUT_TXT)
print(" -", OUT_CSV)
