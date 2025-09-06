# face_cluster_characters.py
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# --- مسیرها ---
ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")
FRAMES_DIR = ROOT / r"mvp_outputs\vision\frames"      # خروجی اسکریپت vision_keyframe_blip_yolo.py
OUT_DIR    = ROOT / r"mvp_outputs\faces"
GALLERY_DIR = ROOT / r"reference_gallery"            # اختیاری: گالری مرجع: هر شخصیت یک پوشه

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- تنظیمات ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECT_MIN_PROB = 0.90
DBSCAN_EPS      = 0.45    # برای metric='cosine' (قابل تنظیم)
DBSCAN_MIN_SAMPLES = 2
SAVE_TOP_PER_CLUSTER = True

# --- مدل‌ها ---
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# --- کمکی ---
scene_re = re.compile(r"scene_(\d+)_")
def parse_scene_id(name: str) -> int:
    m = scene_re.search(name)
    return int(m.group(1)) if m else -1

def extract_faces(img: Image.Image):
    # برمی‌گرداند: faces(tensor Nx3x160x160)، probs(N), boxes(Nx4)
    boxes, probs = mtcnn.detect(img)
    faces = None
    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)
    return faces, probs, boxes

def embed_faces(face_tensors):
    with torch.no_grad():
        face_tensors = face_tensors.to(DEVICE)
        embs = resnet(face_tensors).cpu().numpy()
    return embs

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def load_gallery_embeddings():
    """
    ساخت گالری مرجع اختیاری:
    reference_gallery/
        Tanya/*.jpg
        Joseph/*.jpg
        Paul/*.jpg
    """
    gallery = []  # list of (name, emb)
    if not GALLERY_DIR.exists():
        return gallery
    for person_dir in GALLERY_DIR.iterdir():
        if not person_dir.is_dir(): 
            continue
        embs = []
        for img_path in person_dir.glob("*.*"):
            try:
                img = Image.open(img_path).convert("RGB")
                faces, probs, boxes = extract_faces(img)
                if faces is None or len(faces)==0: 
                    continue
                # بهترین صورت با بیشترین پروب
                best_idx = int(np.argmax(probs))
                f = faces[best_idx].unsqueeze(0)
                e = embed_faces(f)[0]
                embs.append(e)
            except Exception:
                continue
        if embs:
            gallery.append((person_dir.name, np.mean(np.stack(embs, axis=0), axis=0)))
    return gallery

# --- پردازش فریم‌ها ---
all_rows = []   # برای CSV
thumbs = []     # (cluster_id, image)
embeds = []
meta   = []

frames = sorted(FRAMES_DIR.glob("*.jpg"))
if not frames:
    raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")

print(f"Device: {DEVICE} | Frames: {len(frames)}")

for fp in frames:
    img = Image.open(fp).convert("RGB")
    faces, probs, boxes = extract_faces(img)
    scene_id = parse_scene_id(fp.name)

    if faces is None or len(faces)==0:
        continue

    # فیلتر بر اساس پروب
    keep = [i for i,p in enumerate(probs) if (p is not None and p >= DETECT_MIN_PROB)]
    if not keep:
        continue

    faces_kept = faces[keep]
    boxes_kept = np.array(boxes)[keep]
    probs_kept = np.array(probs)[keep]

    # embedding
    embs = embed_faces(faces_kept)
    start_idx = len(embeds)
    embeds.append(embs)
    for i, (b, p) in enumerate(zip(boxes_kept, probs_kept)):
        x1,y1,x2,y2 = map(int, b.tolist())
        all_rows.append({
            "scene_id": scene_id,
            "frame_file": str(fp),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "prob": float(p),
            "emb_index": start_idx + i
        })
        # برای ساخت گالری کوچک
        crop = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        thumbs.append(crop)
    meta.extend([(scene_id, str(fp))] * len(faces_kept))

if not embeds:
    raise RuntimeError("No faces detected with sufficient probability.")

embeds = np.concatenate(embeds, axis=0)
# نرمال‌سازی برای کساین
embeds_norm = normalize(embeds)

# --- کلاسترینگ ---
clt = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine')
labels = clt.fit_predict(embeds_norm)

# نگاشت به نام‌ها با گالری (اختیاری)
gallery = load_gallery_embeddings()
cluster_to_name = {}
if gallery:
    # centroid هر کلاستر را محاسبه کن
    for cid in sorted(set(labels)):
        if cid == -1: 
            continue  # noise
        idxs = np.where(labels == cid)[0]
        centroid = embeds[idxs].mean(axis=0)
        # بهترین تطابق گالری
        best_name, best_sim = None, -1.0
        for name, gemb in gallery:
            sim = cosine_sim(centroid, gemb)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_sim >= 0.55:  # آستانه تطابق (قابل تنظیم)
            cluster_to_name[cid] = best_name

# --- خروجی CSV با کلاستر ---
for i, row in enumerate(all_rows):
    row["cluster_id"] = int(labels[row["emb_index"]])
    row["identity"] = cluster_to_name.get(row["cluster_id"], "")

df_det = pd.DataFrame(all_rows)
df_det = df_det.sort_values(["scene_id", "frame_file"]).reset_index(drop=True)
OUT_CSV = OUT_DIR / "face_detections.csv"
df_det.to_csv(OUT_CSV, index=False)

# --- خلاصه حضور شخصیت‌ها در هر صحنه ---
summary_rows = []
for scene_id, group in df_det.groupby("scene_id"):
    counts = group["cluster_id"].value_counts().to_dict()
    ids = sorted(k for k in counts.keys() if k != -1)
    names = [cluster_to_name.get(k, f"char_{k}") for k in ids]
    summary_rows.append({
        "scene_id": scene_id,
        "clusters": ",".join(map(str, ids)),
        "names": ",".join(names),
        "noise_faces": int(counts.get(-1, 0)),
        "total_faces": int(len(group))
    })

df_sum = pd.DataFrame(summary_rows).sort_values("scene_id")
OUT_SUM = OUT_DIR / "scene_characters.csv"
df_sum.to_csv(OUT_SUM, index=False)

# --- ذخیره نمایه هر کلاستر (یک تصویر نمونه) ---
if SAVE_TOP_PER_CLUSTER:
    gallery_dir = OUT_DIR / "clusters_gallery"
    gallery_dir.mkdir(exist_ok=True, parents=True)
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = int(np.where(labels == cid)[0][0])
        img = thumbs[idx]
        name = cluster_to_name.get(cid, f"char_{cid}")
        cv2.imwrite(str(gallery_dir / f"{cid:02d}_{name}.jpg"), img)

print("Saved:")
print(" -", OUT_CSV)
print(" -", OUT_SUM)
print(" -", OUT_DIR / "clusters_gallery (thumbnails)")
