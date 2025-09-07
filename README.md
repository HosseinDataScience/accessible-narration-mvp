\# Accessible Narration MVP



Make movies more accessible for blind and low-vision audiences by auto-generating concise, scene-aware audio descriptions.  

This MVP fuses \*\*speech (Whisper)\*\*, \*\*environmental audio (PANNs)\*\*, \*\*face \& pose/emotion (FaceNet/MediaPipe/DeepFace)\*\*, and \*\*vision (YOLO + BLIP)\*\* with \*\*BigQuery AI\*\* to build a unified scene timeline and generate narration “stingers” that play between dialogue.



---



\## What this repo contains

\- `vision\_keyframe\_blip\_yolo.py` – extracts keyframes and BLIP captions + YOLO objects

\- `panns\_tag\_other.py` – sound events on the `other.wav` stem (Demucs output)

\- `voice\_cluster\_speakers.py` – speaker embeddings + clustering (SpeechBrain + HDBSCAN)

\- `face\_cluster\_characters.py` – character discovery from faces (FaceNet-PyTorch)

\- `pose\_emotion\_on\_frames.py` – skeleton/pose and basic emotions (MediaPipe + DeepFace)

\- `merge\_whisper\_panns.py` – align Whisper transcript with SFX events into one timeline

\- `build\_accessible\_narration\_en.py` – build the final English narration file

\- `bigquery/queries.sql` – sample \*\*BigQuery AI\*\* SQL showing generative + vector search

\- `requirements.txt` – baseline dependencies to reproduce the pipeline



> Large artifacts (video, stems, frames, outputs) are intentionally ignored by Git with `.gitignore`.  

> Add your own short clip (under 50MB) to test locally, or host larger assets in Cloud Storage.



---



\## Quickstart (local)

1\. Create a Python 3.11 venv and install:

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt



