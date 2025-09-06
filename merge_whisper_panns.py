# merge_whisper_panns.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re

ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")

# ورودی‌ها
WHISPER_DIR = ROOT / r"mvp_outputs\transcripts"
PANNS_EVENTS_CSV = ROOT / r"mvp_outputs\events\panns_other_events.csv"
PANNS_TOP_CSV    = ROOT / r"mvp_outputs\events\panns_other_clip_top20.csv"

# خروجی‌ها
OUT_DIR = ROOT / r"mvp_outputs\merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIALOGUE_CSV = OUT_DIR / "vocals_segments.csv"
MERGED_CSV   = OUT_DIR / "unified_timeline.csv"

def find_whisper_json(transcripts_dir: Path) -> Path:
    # یکی از *.json های خروجی را برمی‌داریم (مثلاً vocals.json یا vocals.wav.json)
    cands = sorted(transcripts_dir.glob("*.json"))
    if not cands:
        raise FileNotFoundError(f"No Whisper JSON found in: {transcripts_dir}")
    # اگر چندتا بود، اولی را می‌گیریم؛ می‌تونی نام دقیق فایل را اینجا هاردکد هم بکنی
    return cands[0]

def load_whisper_segments(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # انتظار داریم data["segments"] موجود باشد
    segs = data.get("segments", [])
    rows = []
    for s in segs:
        start = float(s.get("start", 0.0))
        end   = float(s.get("end",   start))
        text  = s.get("text", "").strip()
        # پاکسازی فاصله‌های اضافی
        text = re.sub(r"\s+", " ", text)
        if text:
            rows.append({"start_s": start, "end_s": end, "text": text})
    df = pd.DataFrame(rows)
    if df.empty:
        # اگر چیزی نبود، یک جدول خالی هم بسازیم
        df = pd.DataFrame(columns=["start_s","end_s","text"])
    return df

def load_panns_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # تضمین ستون‌ها
    for c in ["start_s","end_s","label","mean_score"]:
        if c not in df.columns:
            df[c] = np.nan
    # مرتب‌سازی
    df = df.sort_values(["start_s","end_s"], ignore_index=True)
    return df

def simplify_label(lbl: str) -> str:
    """نام رویدادهای PANNs را کمی خلاصه/نرمال می‌کند تا برای روایت‌گر بهتر شود."""
    if not isinstance(lbl, str):
        return "sound"
    l = lbl.lower()
    # چند نگاشت پرکاربرد
    mapping = {
        "rain": "rain",
        "thunder": "thunder",
        "wind": "wind",
        "fire": "fire",
        "crying": "crying",
        "scream": "scream",
        "footsteps": "footsteps",
        "door": "door",
        "gunshot": "gunshot",
        "explosion": "explosion",
        "engine": "engine",
        "car": "car",
        "water": "water",
        "bird": "bird",
        "dog": "dog",
        "cat": "cat",
        "applause": "applause",
        "laughter": "laughter",
        "music": "music",
        "heartbeat": "heartbeat",
        "phone": "phone",
        "clock": "clock",
        "keyboard": "keyboard",
        "ambient": "ambient",
        "crowd": "crowd",
    }
    for k, v in mapping.items():
        if k in l:
            return v
    # اگر چیزی نخورد:
    return lbl

def build_merged_timeline(df_dialogue: pd.DataFrame, df_sfx: pd.DataFrame) -> pd.DataFrame:
    """
    خروجی: جدول یک‌پارچه با ستون‌های:
    kind ∈ {dialogue, sfx}
    start_s, end_s
    content: متن دیالوگ یا خلاصه رویداد
    extra: فیلد اختیاری (مثلاً label/score)
    """
    # ردیف‌های دیالوگ
    dlg = pd.DataFrame({
        "kind": "dialogue",
        "start_s": df_dialogue["start_s"],
        "end_s": df_dialogue["end_s"],
        "content": df_dialogue["text"].map(str),
        "extra": ""  # رزرو
    })
    # ردیف‌های افکت/ambience
    sfx = df_sfx.copy()
    sfx["simple_label"] = sfx["label"].apply(simplify_label)
    sfx["duration"] = (sfx["end_s"] - sfx["start_s"]).clip(lower=0)
    sfx["content"] = sfx.apply(lambda r: f"{r['simple_label']} (≈{r['duration']:.1f}s)", axis=1)
    sfx_out = pd.DataFrame({
        "kind": "sfx",
        "start_s": sfx["start_s"],
        "end_s": sfx["end_s"],
        "content": sfx["content"],
        "extra": sfx.apply(lambda r: f"label={r['label']};score={r['mean_score']:.2f}", axis=1)
    })

    merged = pd.concat([dlg, sfx_out], ignore_index=True)
    merged = merged.sort_values(["start_s", "end_s", "kind"], ignore_index=True)
    return merged

def main():
    whisper_json = find_whisper_json(WHISPER_DIR)
    print(f"Using Whisper JSON: {whisper_json}")

    df_dialogue = load_whisper_segments(whisper_json)
    df_dialogue.to_csv(DIALOGUE_CSV, index=False)
    print(f"Saved dialogue segments -> {DIALOGUE_CSV}")

    df_sfx = load_panns_events(PANNS_EVENTS_CSV)

    merged = build_merged_timeline(df_dialogue, df_sfx)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"Saved unified timeline -> {MERGED_CSV}")

    # (اختیاری) چاپ چند ردیف اول برای مشاهده سریع
    print("\nPreview:")
    print(merged.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
