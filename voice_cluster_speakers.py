# voice_cluster_speakers.py
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import webrtcvad
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import soundfile as sf

from speechbrain.pretrained import EncoderClassifier

# ---------- تنظیم مسیرها ----------
ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")
VOCALS_WAV = ROOT / r"separated\htdemucs\audio\vocals.wav"
MERGED_DIR = ROOT / r"mvp_outputs\merged"
TRANS_DIR = ROOT / r"mvp_outputs\transcripts"
OUT_DIR = ROOT / r"mvp_outputs\diarization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_SEG = MERGED_DIR / "vocals_segments.csv"     # خروجی merge_whisper_panns.py
JSON_FALLBACK = TRANS_DIR / "vocals.json"        # خروجی whisper در صورت نبود CSV


# ---------- کمکـی: لود و ریسـَمپل ----------
def load_audio_mono(path: Path, target_sr: int = 16000):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr  # (samples,), sr


# ---------- خواندن سگـمنت‌ها ----------
def read_segments() -> pd.DataFrame:
    if CSV_SEG.exists():
        df = pd.read_csv(CSV_SEG)
        # انتظار ستون‌های: start_s, end_s, text
        need = {"start_s", "end_s", "text"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        return df[["start_s", "end_s", "text"]].copy()

    #Fallback: parse Whisper JSON
    if not JSON_FALLBACK.exists():
        raise FileNotFoundError("No segments CSV or JSON found.")
    with open(JSON_FALLBACK, "r", encoding="utf-8") as f:
        j = json.load(f)
    segs = []
    for seg in j.get("segments", []):
        segs.append({
            "start_s": float(seg.get("start", 0.0)),
            "end_s": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip()
        })
    return pd.DataFrame(segs)


# ---------- VAD روی سگمنت (اختیاری ولی کمک‌کننده) ----------
def apply_vad_to_segment(wav_16k: torch.Tensor, sr: int, start_s: float, end_s: float,
                         mode: int = 2, frame_ms: int = 30) -> np.ndarray:
    """
    wav_16k: (samples,)
    خروجی: نمونه‌های تمیزشده (numpy float32)؛ اگر چیزی نماند، None
    """
    assert sr == 16000, "VAD expects 16k"
    vad = webrtcvad.Vad(mode)  # 0-3 (3 = سخت‌گیرتر)
    frame_len = int(sr * frame_ms / 1000)
    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    x = wav_16k[start_i:end_i].clone().detach().cpu().numpy().astype(np.float32)

    if len(x) < frame_len:
        return None

    # PCM 16-bit لازم داریم
    pcm16 = (np.clip(x, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

    voiced_chunks = []
    num_frames = len(x) // frame_len
    for i in range(num_frames):
        chunk = pcm16[i*frame_len*2:(i+1)*frame_len*2]  # *2 چون int16
        is_speech = vad.is_speech(chunk, sample_rate=sr)
        if is_speech:
            seg = x[i*frame_len:(i+1)*frame_len]
            voiced_chunks.append(seg)

    if not voiced_chunks:
        return None

    y = np.concatenate(voiced_chunks, axis=0)
    # اگر خیلی کوتاه بود، برگردون؛ بهتر از هیچی
    if len(y) < int(0.25 * sr):
        return None
    return y.astype(np.float32)


# ---------- استخراج بردار گوینده ----------
def build_embeddings(wav_16k: torch.Tensor, sr: int, df: pd.DataFrame,
                     use_vad=True) -> tuple[np.ndarray, list[dict]]:
    """
    برای هر سطر df یک embedding می‌سازد.
    خروجی:
      - E: آرایه‌ی (N, D) از embedding ها
      - meta: لیستی از دیکشنری با اطلاعات سگمنت
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(OUT_DIR / "ecapa_cache")
    )

    embs = []
    meta = []

    for idx, row in df.iterrows():
        s, e, text = float(row["start_s"]), float(row["end_s"]), str(row["text"])
        dur = max(0.0, e - s)
        if dur < 0.25:
            # خیلی کوتاه، رد می‌کنیم
            continue

        if use_vad:
            y = apply_vad_to_segment(wav_16k, sr, s, e, mode=2, frame_ms=30)
        else:
            start_i = int(s * sr); end_i = int(e * sr)
            y = wav_16k[start_i:end_i].clone().detach().cpu().numpy().astype(np.float32)

        if y is None or len(y) < int(0.25 * sr):
            continue

        # به تنسور برگردان
        seg = torch.from_numpy(y).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = classifier.encode_batch(seg).squeeze(0).squeeze(0).cpu().numpy()  # (192,)
        embs.append(emb)
        meta.append({"start_s": s, "end_s": e, "text": text, "duration": dur})

    if not embs:
        raise RuntimeError("No usable segments for embeddings (maybe all too short?).")

    return np.stack(embs, axis=0), meta


# ---------- خوشه‌بندی ----------
def cluster_embeddings(E: np.ndarray, min_cluster_size=3):
    scaler = StandardScaler()
    Z = scaler.fit_transform(E)

    # کمی کاهش بُعد (پایدارتر می‌شود)
    pca = PCA(n_components=min(50, Z.shape[1]))
    Zp = pca.fit_transform(Z)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                metric='euclidean',
                                cluster_selection_epsilon=0.0,
                                cluster_selection_method='eom',
                                prediction_data=True)
    labels = clusterer.fit_predict(Zp)  # -1 = نویز / تک‌قطعه‌ای
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    return labels, probs


# ---------- خروجی نهایی ----------
def save_results(meta: list[dict], labels: np.ndarray, probs: np.ndarray, out_csv: Path):
    # نگاشت label عددی به اسم گوینده یکنواخت
    uniq = [l for l in sorted(np.unique(labels)) if l != -1]
    mapping = {l: f"SPK_{i:02d}" for i, l in enumerate(uniq)}  # 0,1,2,...
    # برای -1 هم یک نام رزرو کنیم
    mapping[-1] = "SPK_UNKNOWN"

    rows = []
    for m, lab, pr in zip(meta, labels, probs):
        rows.append({
            "start_s": round(float(m["start_s"]), 3),
            "end_s": round(float(m["end_s"]), 3),
            "duration_s": round(float(m["duration"]), 3),
            "text": m["text"],
            "cluster_id": int(lab),
            "speaker": mapping[int(lab)],
            "cluster_confidence": round(float(pr), 3),
        })
    df = pd.DataFrame(rows).sort_values(["start_s", "end_s"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


# ---------- بریدن نمونه‌های صوتی هر گوینده (اختیاری) ----------
def export_snippets_per_speaker(wav_16k: torch.Tensor, sr: int, df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for spk in sorted(df["speaker"].unique()):
        spk_dir = out_dir / spk
        spk_dir.mkdir(exist_ok=True)
        sdf = df[df["speaker"] == spk]
        for i, r in sdf.iterrows():
            s = int(r["start_s"] * sr); e = int(r["end_s"] * sr)
            y = wav_16k[s:e].cpu().numpy().astype(np.float32)
            if len(y) < int(0.2 * sr):  # خیلی کوتاه، رد
                continue
            sf.write(str(spk_dir / f"{r['start_s']:.2f}-{r['end_s']:.2f}.wav"), y, sr)


def main():
    if not VOCALS_WAV.exists():
        raise FileNotFoundError(f"Missing vocals wav: {VOCALS_WAV}")

    print("Loading segments…")
    df_segments = read_segments()
    print(f"Total text segments: {len(df_segments)}")

    print("Loading audio…")
    wav16, sr = load_audio_mono(VOCALS_WAV, target_sr=16000)

    print("Building speaker embeddings…")
    E, meta = build_embeddings(wav16, sr, df_segments, use_vad=True)
    print(f"Embeddings built: {len(meta)}")

    print("Clustering with HDBSCAN…")
    labels, probs = cluster_embeddings(E, min_cluster_size=3)
    print(f"Clusters found: {len([l for l in np.unique(labels) if l!=-1])}, noise: {(labels==-1).sum()}")

    out_csv = OUT_DIR / "segments_with_speakers.csv"
    df_out = save_results(meta, labels, probs, out_csv)
    print(f"Saved -> {out_csv}")

    # (اختیاری) برش نمونه‌های صوتی هر گوینده برای بررسی دستی
    export_snippets_per_speaker(wav16, sr, df_out, OUT_DIR / "snippets_by_speaker")
    print("Done.")


if __name__ == "__main__":
    main()
