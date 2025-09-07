
# Accessible Narration MVP

Make movies more accessible for blind and low-vision audiences by auto-generating concise, scene-aware audio descriptions.  

This MVP fuses **speech (Whisper)**, **environmental audio (PANNs)**, **face & pose/emotion (FaceNet / MediaPipe / DeepFace)**, and **vision (YOLO + BLIP)** with **BigQuery AI** to build a unified scene timeline and generate narration “stingers” that play naturally between dialogue.

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
- `lady_clip.mp4` – **40MB demo video** provided so judges/users can run the pipeline end-to-end without extra setup  
- `run_demo_local.py` – one-click runner to process the demo clip and save narration outputs  

> Large artifacts (long videos, audio stems, frames, outputs) are intentionally ignored by Git with `.gitignore`.  
> For extended testing, you can replace `lady_clip.mp4` with your own short clip (under 50MB) or host larger media in Cloud Storage.

---

## Quickstart (local)

1. Clone the repo and enter the project folder:
   ```bash
   git clone https://github.com/HosseinDataScience/accessible-narration-mvp.git
   cd accessible-narration-mvp
     ```

2. Create a Python 3.11 virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the full demo pipeline on the included 40MB sample video:

   ```bash
   python run_demo_local.py
   ```

4. Outputs will be saved in the `mvp_outputs/` folder (transcripts, vision CSV/JSON, and final narration text).

---

## Roadmap

This MVP currently focuses on generating audio narration for blind and low-vision users.
In future updates, we plan to expand accessibility features, including:

* **Scene context from costume & set design**: using clothing and environment recognition to create richer mental images.
* **Braille output option**: instead of narration audio, output text could be delivered to refreshable Braille devices.
* **Sign language support for deaf audiences**: a multimodal extension where a face icon with lip and hand gestures conveys the story in sign language.

These directions aim to make cinema more inclusive for people with different accessibility needs.

```

---