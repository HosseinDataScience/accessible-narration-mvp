# Accessible Narration MVP

Make movies more accessible for blind and low-vision audiences by auto-generating concise, scene-aware audio descriptions.  

This MVP fuses **speech (Whisper)**, **environmental audio (PANNs)**, **face & pose/emotion (FaceNet / MediaPipe / DeepFace)**, and **vision (YOLO + BLIP)** with **BigQuery AI** to build a unified scene timeline and generate narration “stingers” that play between dialogue.

---

## What this repo contains

- `vision_keyframe_blip_yolo.py` – extracts keyframes and BLIP captions + YOLO objects  
- `panns_tag_other.py` – sound events on the `other.wav` stem (Demucs output)  
- `voice_cluster_speakers.py` – speaker embeddings + clustering (SpeechBrain + HDBSCAN)  
- `face_cluster_characters.py` – character discovery from faces (FaceNet-PyTorch)  
- `pose_emotion_on_frames.py` – skeleton/pose and basic emotions (MediaPipe + DeepFace)  
- `merge_whisper_panns.py` – align Whisper transcript with SFX events into one timeline  
- `build_accessible_narration_en.py` – build the final English narration file  
- `bigquery/queries.sql` – sample **BigQuery AI** SQL showing generative + vector search  
- `requirements.txt` – baseline dependencies to reproduce the pipeline  

> Large artifacts (video, stems, frames, outputs) are intentionally ignored by Git with `.gitignore`.  
> Add your own short clip (under 50MB) to test locally, or host larger assets in Cloud Storage.

---

## Quickstart (local)

1. Create a Python 3.11 virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
