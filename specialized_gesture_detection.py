import torch
import open_clip
import cv2
import os
import time
import json
import numpy as np
from PIL import Image
from datetime import datetime
import threading
import queue

# Load CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'ViT-L-14-quickgelu', pretrained='openai'
)
model = model.to(device)

# Define action categories
primary_classes = {
    "violence": ["violence"],
    "running": ["running", "sprinting", "jogging"],
    "falling": ["a person is unexpectedly falling down",
        "a person is collapsing to the ground",
        "a person is tripping and falling forward",
        "a person loses balance and falls backward"],
    "normal": ["normal","people are walking","a person is sitting calmly",
        "a person is walking at a normal pace",
        "a quiet and empty room",
        "a person is talking casually",
        "a person with no movement or activity",
        "a person is strolling slowly",
        "a person is walking with normal steps",
        "a person is walking in a relaxed manner",
        "a person is walking at a moderate speed",
        "a person's expression and action is not visible",
        "unknown action"]
}
secondary_classes = {
    "punching": ["punching", "throwing a punch"],
    "hitting": ["hitting", "slapping", "beating", "kicking"],
    "shoving": ["pushing", "shoving"],
    "shouting": ["shouting", "yelling", "screaming"]
}

# Tokenize primary labels
primary_labels = []
label_map = {}
for category, descriptions in primary_classes.items():
    for desc in descriptions:
        primary_labels.append(desc)
        label_map[desc] = category
primary_text = open_clip.tokenize(primary_labels).to(device)

# Create base folders
video_folder = "video4_clips"
labelled_folder = "labelled4_videos"
os.makedirs(video_folder, exist_ok=True)
os.makedirs(labelled_folder, exist_ok=True)

# Queue for video processing
video_queue = queue.Queue()
stop_event = threading.Event()

# Capture thread (producer)
def capture_videos():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20

    while not stop_event.is_set():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_folder, f"clip_{timestamp}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        start_time = time.time()
        while (time.time() - start_time) < 5:  # Record for 5 seconds
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            out.write(frame)
        
        out.release()
        video_queue.put(video_path)  # Add to queue
        print(f"📹 Saved: {video_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    print("🎥 Stopped video capture.")

# Processing thread (consumer)
"""
def process_videos():
    while not stop_event.is_set() or not video_queue.empty():
        try:
            video_path = video_queue.get(timeout=1)
        except queue.Empty:
            continue  # If queue is empty, continue checking
        
        # Get current date folder
        date_str = datetime.now().strftime("%Y-%m-%d")
        daily_folder = os.path.join(labelled_folder, date_str)
        os.makedirs(daily_folder, exist_ok=True)

        # Define paths
        timestamp = os.path.basename(video_path).replace("clip_", "labelled_")
        labelled_video_path = os.path.join(daily_folder, timestamp)
        log_file_path = os.path.join(daily_folder, "detection_log.txt")

        cap_clip = cv2.VideoCapture(video_path)
        frame_width = int(cap_clip.get(3))
        frame_height = int(cap_clip.get(4))
        fps = 20
        out_clip = cv2.VideoWriter(labelled_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Dictionary to store detections
        detection_counts = {"normal": 0, "running": 0, "violence": 0, "falling":0}
        action_counts = {action: 0 for action in secondary_classes.keys()}

        while cap_clip.isOpened():
            ret, frame = cap_clip.read()
            if not ret:
                break
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(primary_text)
            
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_description = primary_labels[similarity.argmax()]
            best_category = label_map[best_description]
            best_score = similarity.max().item()
            
            thresholds = {"normal": 0.95, "running": 0.95, "violence": 0.95, "falling": 0.95}
            detected_class = "normal"
            if best_score >= thresholds.get(best_category, 0):
                detected_class = best_category
            
            detection_counts[detected_class] += 1
            action_text = detected_class
            
            if detected_class == "violence":
                action_scores = {}
                for action, descriptions in secondary_classes.items():
                    action_texts = open_clip.tokenize(descriptions).to(device)
                    action_features = model.encode_text(action_texts)
                    similarity_action = (image_features @ action_features.T).softmax(dim=-1)
                    action_scores[action] = similarity_action.max().item()
                
                best_action = max(action_scores, key=action_scores.get)
                action_confidence = action_scores[best_action]
                if action_confidence >= 0.65:
                    action_text = best_action
                    action_counts[best_action] += 1  # Update action count
            
            label_text = f"{detected_class.upper()} - {action_text} ({best_score:.2f})"
            color = (0, 255, 0) if detected_class == "normal" else (255, 0, 0) if detected_class == "running" else (0, 0, 255) if detected_class=="violence" else (255,165,0)
            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            out_clip.write(frame)
        
        cap_clip.release()
        out_clip.release()

        # Load existing log file if it exists
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                try:
                    log_data = json.load(log_file)
                except json.JSONDecodeError:
                    log_data = {}
        else:
            log_data = {}

        # Save new detection results
        log_data[timestamp] = {
            "primary_counts": detection_counts,  # Store primary detections
            "secondary_counts": action_counts   # Store all secondary counts (even if 0)
        }

        with open(log_file_path, "w") as log_file:
            json.dump(log_data, log_file, indent=4)

        print(f"✅ Processed & Logged: {labelled_video_path}")"""
def process_videos():
    while not stop_event.is_set() or not video_queue.empty():
        try:
            video_path = video_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        cap_clip = cv2.VideoCapture(video_path)
        frame_width = int(cap_clip.get(3))
        frame_height = int(cap_clip.get(4))
        fps = 20

        # Temporary output path (won't save if not needed)
        temp_output_path = "temp_output.mp4"
        out_clip = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        detection_counts = {"normal": 0, "running": 0, "violence": 0, "falling": 0}
        action_counts = {action: 0 for action in secondary_classes.keys()}
        frames = []

        while cap_clip.isOpened():
            ret, frame = cap_clip.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(primary_text)

            similarity = (image_features @ text_features.T).softmax(dim=-1)
            best_description = primary_labels[similarity.argmax()]
            best_category = label_map[best_description]
            best_score = similarity.max().item()

            thresholds = {"normal": 0.85, "running": 0.85, "violence": 0.85, "falling": 0.85}
            detected_class = "normal"
            if best_score >= thresholds.get(best_category, 0):
                detected_class = best_category

            detection_counts[detected_class] += 1
            action_text = detected_class

            if detected_class == "violence":
                action_scores = {}
                for action, descriptions in secondary_classes.items():
                    action_texts = open_clip.tokenize(descriptions).to(device)
                    action_features = model.encode_text(action_texts)
                    similarity_action = (image_features @ action_features.T).softmax(dim=-1)
                    action_scores[action] = similarity_action.max().item()

                best_action = max(action_scores, key=action_scores.get)
                action_confidence = action_scores[best_action]
                if action_confidence >= 0.65:
                    action_text = best_action
                    action_counts[best_action] += 1

            label_text = f"{detected_class.upper()} - {action_text} ({best_score:.2f})"
            color = (0, 255, 0) if detected_class == "normal" else (255, 0, 0) if detected_class == "running" else (0, 0, 255) if detected_class == "violence" else (255, 165, 0)
            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            out_clip.write(frame)

        cap_clip.release()
        out_clip.release()

        # Check if the clip should be saved (contains at least 1 non-normal action)
        if detection_counts["violence"] > 0 or detection_counts["running"] > 0 or detection_counts["falling"] > 0:
            # Save the clip in labelled folder
            date_str = datetime.now().strftime("%Y-%m-%d")
            daily_folder = os.path.join(labelled_folder, date_str)
            os.makedirs(daily_folder, exist_ok=True)

            timestamp = os.path.basename(video_path).replace("clip_", "labelled_")
            labelled_video_path = os.path.join(daily_folder, timestamp)
            os.rename(temp_output_path, labelled_video_path)

            # Save detection logs
            log_file_path = os.path.join(daily_folder, "detection_log.txt")
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as log_file:
                    try:
                        log_data = json.load(log_file)
                    except json.JSONDecodeError:
                        log_data = {}
            else:
                log_data = {}

            log_data[timestamp] = {
                "primary_counts": detection_counts,
                "secondary_counts": action_counts
            }

            with open(log_file_path, "w") as log_file:
                json.dump(log_data, log_file, indent=4)

            print(f"✅ Processed & Saved: {labelled_video_path}")
        else:
            # Remove temporary video if all actions were normal
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            print(f"⚠️ Skipped (All Normal): {video_path}")


# Start threads
capture_thread = threading.Thread(target=capture_videos)
process_thread = threading.Thread(target=process_videos)

capture_thread.start()
process_thread.start()

# Wait for capture to stop
capture_thread.join()

# Ensure all videos in the queue are processed before exiting
stop_event.set()
process_thread.join()

cv2.destroyAllWindows()
print("🎉 Processing completed.")
