# Video Action Recognition using CLIP

This project captures live video clips from a webcam, processes them using OpenAI's CLIP model to classify human actions, and labels the detected actions on the video frames. The processed videos are saved with corresponding detection logs.

## Features
- Captures 5-second video clips from a webcam.
- Uses CLIP (ViT-L-14-quickgelu) to classify actions.
- Categorizes actions into primary (e.g., normal, running, violence) and secondary (e.g., punching, shoving, shouting).
- Saves labeled videos and logs detection results.
- Runs in a multi-threaded setup with separate producer and consumer threads.

## Installation
Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```sh
pip install torch torchvision torchaudio open-clip-torch opencv-python numpy pillow
```

## Usage
Run the script to start capturing and processing videos:

```sh
python specialized_gesture_detection.py
```

### Key Directories
- `video4_clips/`: Stores captured raw video clips.
- `labelled4_videos/`: Contains processed videos with action labels, organized by date.

### Log File Structure
Each processed video has a corresponding log file (`detection_log.txt`), storing action counts in JSON format:

```json
{
    "labelled_20240510_123456.mp4": {
        "primary_counts": {"normal": 10, "running": 5, "violence": 2,"falling":1},
        "secondary_counts": {"punching": 1, "shoving": 1}
    }
}
```

## Stopping the Script
To stop video capturing, press `q`.

## Notes
- Ensure your camera is accessible before running the script.
- Adjust classification thresholds if needed in the `thresholds` dictionary.

## License
This project is open-source and free to use.

