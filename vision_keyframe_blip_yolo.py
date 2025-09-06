# vision_keyframe_blip_yolo.py
from pathlib import Path
import csv
import json
import math
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")
VIDEO = ROOT / "lady_clip.mp4"
# اگر قبلاً با PySceneDetect ران کردی، این CSV را بده.
# اگر نداشتی، اسکریپت خودش هر 6 ثانیه یک صحنه فرض می‌گیرد.
SCENES_CSV = ROOT / "mvp_outputs" / "scenes" / "scene_list.csv"
OUT_DIR = ROOT / "mvp_outputs" / "vision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ابزار ----------
def read_scenes_from_csv(path: Path):
    rows = []
    if path.exists():
        df = pd.read_csv(path)
        # انتظار ستون‌های start/end در ثانیه
        if {"start_seconds","end_seconds"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append((float(r["start_seconds"]), float(r["end_seconds"])))
    return rows

def fallback_uniform_scenes(video_path: Path, seg=6.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = frames / max(fps, 1e-6)
    cap.release()
    t = 0.0
    out = []
    while t < dur:
        out.append((t, min(t+seg, dur)))
        t += seg
    return out

def get_frame_at(video_path: Path, t_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec*1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    # BGR→RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def camera_emphasis(frame_rgb: np.ndarray):
    """
    امتیاز ساده‌ی «تأکید تصویری»:
    - سهم جسم/اجسام بزرگ در مرکز (با YOLO boxes)
    - میزان کنتراست/وضوح (لبه‌ها)
    - نسبت نگه‌داشتن شات (user later)
    اینجا فاز۱: فقط تمرکز مرکزی + کنتراست.
    """
    h, w, _ = frame_rgb.shape
    cx0, cy0, cx1, cy1 = int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)
    roi = cv2.cvtColor(frame_rgb[cy0:cy1, cx0:cx1], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(roi, 50, 150)
    edge_density = edges.mean()/255.0  # 0..1
    return float(edge_density)

# ---------- مدل‌ها ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt")  # سبک و سریع، کلاس‌های COCO
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ---------- صحنه‌ها ----------
scenes = read_scenes_from_csv(SCENES_CSV)
if not scenes:
    scenes = fallback_uniform_scenes(VIDEO, seg=6.0)

records = []
frames_dir = OUT_DIR / "frames"
frames_dir.mkdir(exist_ok=True, parents=True)

for idx, (ts, te) in enumerate(scenes, start=1):
    mid = 0.5*(ts+te)
    frame = get_frame_at(VIDEO, mid)
    if frame is None: continue

    # ذخیره فریم
    img_path = frames_dir / f"scene_{idx:03d}_{int(mid*1000):07d}ms.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # BLIP caption
    inputs = blip_proc(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = blip.generate(**inputs, max_new_tokens=40)
    caption = blip_proc.decode(out_ids[0], skip_special_tokens=True).strip()

    # YOLO objects
    y = yolo.predict(frame, conf=0.25, verbose=False)
    objects = []
    center_mass = 0.0
    if y and len(y):
        res = y[0].boxes
        for b in res:
            cls = int(b.cls.item())
            name = yolo.model.names.get(cls, f"id_{cls}")
            conf = float(b.conf.item())
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            w = max(x2-x1, 1.0); h = max(y2-y1, 1.0)
            area = (w*h) / (frame.shape[0]*frame.shape[1])
            cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
            # وزن‌دهی بیشتر به مرکز قاب
            wx = 1.0 - abs((cx/frame.shape[1]) - 0.5)*2
            wy = 1.0 - abs((cy/frame.shape[0]) - 0.5)*2
            center_w = max(0.0, wx*wy)
            center_mass += area*center_w
            objects.append({"label": name, "conf": round(conf,3), "area": round(area,4)})

    emph = camera_emphasis(frame)
    emphasis_score = float(min(1.0, center_mass*0.7 + emph*0.3))

    records.append({
        "scene_id": idx,
        "start_s": round(ts,2),
        "end_s": round(te,2),
        "mid_s": round(mid,2),
        "frame_path": str(img_path),
        "caption_blip": caption,
        "objects_yolo": objects,
        "emphasis": round(emphasis_score,3)
    })

# خروجی‌ها
json_path = OUT_DIR / "scene_vision.jsonl"
with open(json_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

csv_rows = []
for r in records:
    obj_list = ", ".join(sorted({o["label"] for o in r["objects_yolo"]}))
    csv_rows.append([r["scene_id"], r["start_s"], r["end_s"], r["mid_s"], r["emphasis"], obj_list, r["caption_blip"], r["frame_path"]])

csv_path = OUT_DIR / "scene_vision.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scene_id","start_s","end_s","mid_s","emphasis","objects","blip_caption","frame"])
    w.writerows(csv_rows)

print("Saved:")
print(" -", csv_path)
print(" -", json_path)
