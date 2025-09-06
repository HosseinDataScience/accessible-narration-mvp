# pose_emotion_on_frames.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from deepface import DeepFace

ROOT = Path(r"C:\Users\hosse\OneDrive\Desktop\BigQuery AI Challenge")
FRAMES_DIR = ROOT / r"mvp_outputs\vision\frames"
FACE_DET_CSV = ROOT / r"mvp_outputs\faces\face_detections.csv"
OUT_DIR = ROOT / r"mvp_outputs\pose_emotion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSE_CSV = OUT_DIR / "pose_per_frame.csv"
EMO_CSV  = OUT_DIR / "emotion_per_face.csv"
SCENE_SUMMARY = OUT_DIR / "scene_pose_emotion_summary.csv"

mp_pose = mp.solutions.pose

def load_frames():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")
    return frames

def infer_pose_on_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        # چند نقطه کلیدی:
        lm = res.pose_landmarks.landmark
        def p(i): return np.array([lm[i].x, lm[i].y])
        # شانه‌ها، ران‌ها
        l_sh, r_sh = p(mp_pose.PoseLandmark.LEFT_SHOULDER), p(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_hip, r_hip = p(mp_pose.PoseLandmark.LEFT_HIP), p(mp_pose.PoseLandmark.RIGHT_HIP)
        # زاویه تنه (بین خط شانه و ران)
        torso_vec = ((l_sh + r_sh)/2) - ((l_hip + r_hip)/2)
        vert = np.array([0, -1.0])  # محور عمودی به بالا
        cosang = np.dot(torso_vec, vert) / (np.linalg.norm(torso_vec)+1e-8)
        # شاخص ایستاده/خم‌شده:
        uprightness = max(-1.0, min(1.0, cosang))  # -1 خم شده، +1 عمودی
        # فاصله عمودی سر تا لگن → سرنخی از نشسته بودن (نسبی، تقریبی)
        head = p(mp_pose.PoseLandmark.NOSE)  # تقریبی
        torso_len = np.linalg.norm(((l_sh+r_sh)/2 - (l_hip+r_hip)/2))
        head_hip = np.linalg.norm(head - ((l_hip+r_hip)/2))
        # اگر head_hip کوچک و uprightness پایین → نشسته/خمیده
        posture = "standing" if (uprightness > 0.2 and head_hip > torso_len*0.9) else ("sitting/leaning" if uprightness < 0.1 else "mixed")
        return {
            "uprightness": float(uprightness),
            "posture": posture
        }

def crop_face(img_bgr, box):
    x1,y1,x2,y2 = map(int, box)
    h, w = img_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2].copy()

def main():
    frames = load_frames()
    # pose per frame
    pose_rows = []
    for fp in frames:
        img = cv2.imread(str(fp))
        pose_info = infer_pose_on_image(img)
        if pose_info is None:
            pose_rows.append({"frame_file": str(fp), "uprightness": np.nan, "posture": "unknown"})
        else:
            pose_rows.append({"frame_file": str(fp), **pose_info})
    df_pose = pd.DataFrame(pose_rows)
    df_pose.to_csv(POSE_CSV, index=False)

    # emotion per face (با کروپ‌های face_detections.csv)
    if not FACE_DET_CSV.exists():
        raise FileNotFoundError(f"Missing {FACE_DET_CSV}")
    df_faces = pd.read_csv(FACE_DET_CSV)

    emo_rows = []
    grouped = df_faces.groupby("frame_file")
    for frame_path, g in grouped:
        img = cv2.imread(frame_path)
        for _, r in g.iterrows():
            box = [r["x1"], r["y1"], r["x2"], r["y2"]]
            face = crop_face(img, box)
            if face is None:
                continue
            try:
                # DeepFace: 'emotion' → dict از برچسب‌ها و امتیازها
                analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                emo = analysis[0]['dominant_emotion']
                emo_scores = analysis[0]['emotion']
                score = float(emo_scores.get(emo, 0.0))
            except Exception:
                emo, score = "unknown", 0.0
            emo_rows.append({
                "frame_file": frame_path,
                "scene_id": r.get("scene_id", -1),
                "cluster_id": r.get("cluster_id", -1),
                "identity": r.get("identity", ""),
                "dominant_emotion": emo,
                "emotion_score": score
            })
    df_emo = pd.DataFrame(emo_rows)
    df_emo.to_csv(EMO_CSV, index=False)

    # خلاصهٔ صحنه‌ای (pose + emotion)
    # نگاشت فریم→scene از نام فایل (scene_XXX_)
    def parse_scene_id_from_name(name: str):
        import re
        m = re.search(r"scene_(\d+)_", name)
        return int(m.group(1)) if m else -1

    df_pose["scene_id"] = df_pose["frame_file"].apply(lambda s: parse_scene_id_from_name(Path(s).name))
    # غالب احساس هر صحنه
    if not df_emo.empty:
        emo_summary = (df_emo
            .groupby(["scene_id","dominant_emotion"])["emotion_score"].mean()
            .reset_index()
            .sort_values(["scene_id","emotion_score"], ascending=[True, False])
            .groupby("scene_id")
            .head(1)
            .rename(columns={"dominant_emotion":"scene_emotion","emotion_score":"scene_emotion_score"})
        )
    else:
        emo_summary = pd.DataFrame(columns=["scene_id","scene_emotion","scene_emotion_score"])

    # غالب وضعیت بدنی هر صحنه
    if not df_pose.empty:
        pose_summary = (df_pose
            .groupby(["scene_id","posture"])["uprightness"].mean()
            .reset_index()
            .sort_values(["scene_id","uprightness"], ascending=[True, False])
            .groupby("scene_id")
            .head(1)
            .rename(columns={"posture":"scene_posture","uprightness":"scene_uprightness"})
        )
    else:
        pose_summary = pd.DataFrame(columns=["scene_id","scene_posture","scene_uprightness"])

    # ادغام
    df_sum = pd.merge(pose_summary, emo_summary, on="scene_id", how="outer").sort_values("scene_id")
    df_sum.to_csv(SCENE_SUMMARY, index=False)
    print("Saved:")
    print(" -", POSE_CSV)
    print(" -", EMO_CSV)
    print(" -", SCENE_SUMMARY)

if __name__ == "__main__":
    main()
