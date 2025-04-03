import os
import json
import cv2
import torch
import open_clip
import numpy as np
import threading
from queue import Queue
from datetime import datetime
from PIL import Image

# Directories
video_folder = "video8_clips"
labelled_folder = "labelled8_videos"
os.makedirs(video_folder, exist_ok=True)
os.makedirs(labelled_folder, exist_ok=True)

# Processing Queue
video_queue = Queue()

# Load CLIP Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)

# Labels
labels = ["violence", "normal"]
text = open_clip.tokenize(labels).to(device)

# Open Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Unable to access camera")
    exit()

# Video Properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
clip_duration = 5  # 5 seconds per clip

def process_videos():
    """Processes video clips: runs detection and saves labelled video + logs."""
    while True:
        clip_path = video_queue.get()  # Get next clip
        if clip_path is None:
            break  # Exit if None received

        print(f"🔄 Processing: {clip_path}")

        # Read Video
        cap_clip = cv2.VideoCapture(clip_path)
        frame_count = 0
        labelled_path = os.path.join(labelled_folder, os.path.basename(clip_path))
        log_path = os.path.join(labelled_folder, "detection_log.json")

        # Create Video Writer
        out = cv2.VideoWriter(labelled_path, fourcc, fps, (frame_width, frame_height))

        # Load existing detection log
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                detection_log = json.load(f)
        else:
            detection_log = {}

        while cap_clip.isOpened():
            ret, frame = cap_clip.read()
            if not ret:
                break

            # Preprocess Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Encode Image & Text
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text)

            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_label = labels[similarity.argmax()]

            # Add Text Overlay
            cv2.putText(frame, best_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save Frame
            out.write(frame)

            # Update Detection Log
            filename = os.path.basename(clip_path)
            if filename not in detection_log:
                detection_log[filename] = {"violence": 0, "normal": 0}

            detection_log[filename][best_label] += 1
            frame_count += 1

        # Save Detection Log
        with open(log_path, "w") as f:
            json.dump(detection_log, f, indent=4)

        cap_clip.release()
        out.release()
        print(f"✅ Processed & Saved: {labelled_path}")

def record_videos():
    """Continuously records 5s video clips and adds them to the queue."""
    while True:
        clip_start_time = datetime.now()
        clip_timestamp = clip_start_time.strftime("%Y%m%d_%H%M%S")
        clip_path = os.path.join(video_folder, f"clip_{clip_timestamp}.mp4")
        
        clip_out = cv2.VideoWriter(clip_path, fourcc, fps, (frame_width, frame_height))
        print(f"🎬 Recording Clip: {clip_path}")

        while (datetime.now() - clip_start_time).total_seconds() < clip_duration:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break

            clip_out.write(frame)
        
        clip_out.release()
        print(f"📹 Saved: {clip_path}")

        # Add to processing queue
        video_queue.put(clip_path)

# Start Processing Thread
processing_thread = threading.Thread(target=process_videos, daemon=True)
processing_thread.start()

# Start Recording
record_videos()
