# panns_tag_other.py
from pathlib import Path
import os
import numpy as np
import pandas as pd
import librosa

from panns_inference import AudioTagging, SoundEventDetection, labels as PANN_LABELS

# ================== مسیرها ==================
ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")
WAV_PATH = ROOT / r"separated\htdemucs\audio\other.wav"
OUT_DIR = ROOT / r"mvp_outputs\events"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANNS_DIR = Path(os.path.expandvars(r"%USERPROFILE%\panns_data"))
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
AT_CKPT    = PANNS_DIR / "Cnn14_mAP=0.431.pth"                     # clip-level
SED_CKPT   = PANNS_DIR / "Cnn14_DecisionLevelMax_mAP=0.385.pth"    # frame-level

for p in [LABELS_CSV, AT_CKPT, SED_CKPT]:
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {p}")

# ================== تنظیمات ==================
TARGET_SR = 32000
THRESH    = 0.20   # آستانه اطمینان فریم‌ها
MIN_DUR   = 0.60   # حداقل طول رویداد ادغام‌شده (ثانیه)

# ================== توابع کمکی ==================
def load_audio_mono_32k(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)

def rle_merge(df_frames: pd.DataFrame, thresh: float, min_dur: float) -> pd.DataFrame:
    """ادغام بازه‌های پیوسته برای یک برچسب یکسان با RLE ساده."""
    df_f = df_frames[df_frames["score"] >= thresh].copy()
    events = []
    if not df_f.empty:
        cur_label = df_f.iloc[0]["label"]
        cur_start = float(df_f.iloc[0]["start_s"])
        cur_end   = float(df_f.iloc[0]["end_s"])
        cur_scores = [float(df_f.iloc[0]["score"])]

        for _, row in df_f.iloc[1:].iterrows():
            if row["label"] == cur_label and abs(float(row["start_s"]) - cur_end) < 1e-6:
                cur_end = float(row["end_s"])
                cur_scores.append(float(row["score"]))
            else:
                if (cur_end - cur_start) >= min_dur:
                    events.append((cur_start, cur_end, cur_label, float(np.mean(cur_scores))))
                cur_label  = row["label"]
                cur_start  = float(row["start_s"])
                cur_end    = float(row["end_s"])
                cur_scores = [float(row["score"])]

        if (cur_end - cur_start) >= min_dur:
            events.append((cur_start, cur_end, cur_label, float(np.mean(cur_scores))))

    return pd.DataFrame(events, columns=["start_s", "end_s", "label", "mean_score"])\
             .sort_values(["start_s", "end_s"], ignore_index=True)

def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def normalize_at_output(clip_out):
    """
    AudioTagging.inference:
      - dict: {'clipwise_output', 'embedding', 'labels'(اختیاری)}
      - tuple(3): (clipwise_output, embedding, labels)
      - tuple(2): (clipwise_output, embedding)
    -> (clipwise_output: np.ndarray[(B, C)], labels: list[str])
    """
    if isinstance(clip_out, dict):
        clipwise = clip_out["clipwise_output"]
        labels   = clip_out.get("labels", PANN_LABELS)
        return clipwise, labels

    # tuple / list
    n = len(clip_out)
    if n == 3:
        clipwise, _, labels = clip_out
        return clipwise, labels
    elif n == 2:
        clipwise, _ = clip_out
        return clipwise, PANN_LABELS
    else:
        raise ValueError(f"Unexpected AudioTagging output structure: len={n}")

def normalize_sed_output(sed_out):
    """
    SoundEventDetection.inference:
      - dict: {'framewise_output', 'frame_hop', 'labels'(اختیاری)}
      - tuple(3): (framewise_output, frame_hop, labels)
      - tuple(2): (framewise_output, frame_hop)
      - tuple(1): (framewise_output,)
    -> (framewise_output: np.ndarray[(?, ?, ?)], frame_hop: float, labels: list[str])
    """
    DEFAULT_HOP = 0.02  # ~50 FPS

    if isinstance(sed_out, dict):
        framewise = sed_out["framewise_output"]
        hop       = sed_out.get("frame_hop", DEFAULT_HOP)
        labels    = sed_out.get("labels", PANN_LABELS)
        return framewise, float(hop), labels

    n = len(sed_out)
    if n == 3:
        framewise, hop, labels = sed_out
        return framewise, float(hop), labels
    elif n == 2:
        framewise, hop = sed_out
        return framewise, float(hop), PANN_LABELS
    elif n == 1:
        framewise = sed_out[0]
        print(f"[WARN] SED returned only framewise_output; using default frame_hop={DEFAULT_HOP}s")
        return framewise, float(DEFAULT_HOP), PANN_LABELS
    else:
        raise ValueError(f"Unexpected SED output structure: len={n}")

# ================== بارگذاری و آماده‌سازی ==================
if not WAV_PATH.exists():
    raise FileNotFoundError(f"Audio not found: {WAV_PATH}")

waveform = load_audio_mono_32k(WAV_PATH)  # (n_samples,)
audio = waveform[None, :]                 # (1, n_samples) — batch dimension

device = pick_device()
print(f"Device: {device}")

# ================== 1) برچسب‌زنی کلی (Clip-level) ==================
at = AudioTagging(checkpoint_path=str(AT_CKPT), device=device)
clip_out = at.inference(audio)                      # dict یا tuple
clipwise_all, class_names_at = normalize_at_output(clip_out)

# batch اول
clip_scores = clipwise_all[0]                       # (classes,)
clip_labels = class_names_at

df_clip = pd.DataFrame({"label": clip_labels, "score": clip_scores})\
           .sort_values("score", ascending=False).head(20)
clip_csv = OUT_DIR / "panns_other_clip_top20.csv"
df_clip.to_csv(clip_csv, index=False)

# ================== 2) رویدادهای زمان‌مند (Frame-level) ==================
sed = SoundEventDetection(checkpoint_path=str(SED_CKPT), device=device)
sed_out = sed.inference(audio)
framewise_any, frame_hop, class_names_sed = normalize_sed_output(sed_out)

# ---- یکسان‌سازی شکل خروجی SED ----
# حالت‌ها:
# (B, T, C) → می‌گیریم [0, :, :]
# (T, C)    → همان است
# (C,)      → تبدیل به (1, C) و یک فریم فرض می‌کنیم
framewise_any = np.asarray(framewise_any)
if framewise_any.ndim == 3:
    framewise = framewise_any[0]                   # (T, C)
elif framewise_any.ndim == 2:
    framewise = framewise_any                      # (T, C)
elif framewise_any.ndim == 1:
    print("[WARN] SED framewise_output is 1D; treating as single-frame timeline.")
    framewise = framewise_any[None, :]             # (1, C)
else:
    raise ValueError(f"Unexpected SED framewise_output ndim={framewise_any.ndim}")

# احتیاط: اگر طول لیبل‌ها با C هم‌خوانی نداشت، از PANN_LABELS استفاده کن
C = framewise.shape[1]
if len(class_names_sed) != C:
    print(f"[WARN] Label count ({len(class_names_sed)}) != num_classes ({C}); falling back to PANN_LABELS.")
    class_names_sed = PANN_LABELS
    # اگر باز هم هم‌خوان نبود، آخرین چاره: ساخت لیبل‌های اندیسی
    if len(class_names_sed) != C:
        class_names_sed = [f"class_{i}" for i in range(C)]

frames = framewise.shape[0]
starts = np.arange(frames, dtype=float) * frame_hop
ends   = starts + frame_hop

# برچسب غالب هر فریم
top_idx   = framewise.argmax(axis=1)
top_score = framewise[np.arange(frames), top_idx]
top_label = [class_names_sed[i] for i in top_idx]

df_frames = pd.DataFrame({
    "frame":   np.arange(frames),
    "start_s": starts,
    "end_s":   ends,
    "label":   top_label,
    "score":   top_score,
})
frames_csv = OUT_DIR / "panns_other_frames.csv"
df_frames.to_csv(frames_csv, index=False)

# ادغام بازه‌های پیوسته
df_events = rle_merge(df_frames, THRESH, MIN_DUR)
events_csv = OUT_DIR / "panns_other_events.csv"
df_events.to_csv(events_csv, index=False)

print("Saved:")
print(" -", clip_csv)
print(" -", frames_csv)
print(" -", events_csv)
