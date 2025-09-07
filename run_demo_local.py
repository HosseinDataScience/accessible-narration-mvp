# run_demo_local.py
"""
One-click demo runner for Accessible Narration MVP.
Runs a minimal end-to-end pass on lady_clip.mp4 and writes outputs to mvp_outputs/.
Safe to re-run; it skips steps if outputs exist.
"""

from pathlib import Path
import subprocess, sys, shutil

ROOT = Path(__file__).parent
VIDEO = ROOT / "lady_clip.mp4"
OUT   = ROOT / "mvp_outputs"
OUT.mkdir(exist_ok=True)

def run(title, cmd):
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), shell=False)
    if r.returncode != 0:
        print(f"[WARN] Step failed: {title} (code {r.returncode})")
        sys.exit(r.returncode)

# 0) Quick checks
if not VIDEO.exists():
    sys.exit("Demo video not found: lady_clip.mp4")

# 1) Vision keyframes + BLIP + YOLO
vision_csv = OUT / "vision" / "scene_vision.csv"
if not vision_csv.exists():
    run("Vision (keyframes + BLIP + YOLO)",
        [sys.executable, "vision_keyframe_blip_yolo.py"])
else:
    print("[skip] Vision already exists")

# 2) Optional: PANNs SFX over separated stems (requires 'separated/htdemucs/audio/other.wav')
panns_csv = OUT / "events" / "panns_other_events.csv"
if not panns_csv.exists():
    other_wav = ROOT / "separated" / "htdemucs" / "audio" / "other.wav"
    if other_wav.exists():
        run("PANNs sound events", [sys.executable, "panns_tag_other.py"])
    else:
        print("[skip] PANNs (no Demucs stems found)")

# 3) Optional: Faces / pose / emotion
face_csv = OUT / "faces" / "scene_characters.csv"
pose_csv = OUT / "pose_emotion" / "scene_pose_emotion_summary.csv"
if not face_csv.exists():
    run("Face clustering", [sys.executable, "face_cluster_characters.py"])
else:
    print("[skip] Face clustering already exists")
if not pose_csv.exists():
    run("Pose & Emotion", [sys.executable, "pose_emotion_on_frames.py"])
else:
    print("[skip] Pose/Emotion already exists")

# 4) Merge transcript (Whisper) + SFX into unified timeline (if transcript exists)
merged_csv = OUT / "merged" / "unified_timeline.csv"
if not merged_csv.exists():
    vocals_json = OUT / "transcripts" / "vocals.json"
    if vocals_json.exists() or panns_csv.exists():
        run("Merge Whisper + PANNs", [sys.executable, "merge_whisper_panns.py"])
    else:
        print("[skip] Merge (no transcript or SFX)")

# 5) Build English narration (works with whatever is available)
final_txt = OUT / "narration" / "accessible_narration_en.txt"
if not final_txt.exists():
    run("Build narration", [sys.executable, "build_accessible_narration_en.py"])
else:
    print("[skip] Narration already exists")

print("\nDONE ✓  See outputs under mvp_outputs/")
