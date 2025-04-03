# Action Recognition using CLIP

## Overview
This project implements real-time action recognition using OpenAI's CLIP model. It processes webcam frames and classifies detected actions based on predefined descriptions.

## Requirements
Ensure you have the necessary dependencies installed. You can install them using:
```bash
pip install -r requirements.txt
```

## Setup
1. Clone the repository or download the script.
2. Ensure you have Python installed (version 3.7 or later).
3. Install dependencies using the above command.
4. If using a GPU, ensure CUDA is installed and available.

## Usage
To run the script, execute:
```bash
python vit.py
```
Press 'q' to exit the real-time webcam detection.

## Action Descriptions
The following actions are recognized based on descriptions:
- **Punching**: A person is violently punching another person.
- **Kicking**: A person is aggressively kicking another person.
- **Slapping**: A person is slapping another person aggressively.
- **Pushing**: A person is forcefully pushing another person.
- **Shouting**: A person is yelling aggressively.
- **Falling**: A person is falling down.
- **Running**: A person is running away in panic.
- **Normal**: A person is behaving normally.

## Troubleshooting
- **Webcam not opening?** Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- **Low accuracy?** Adjust the confidence threshold in the script.
- **GPU not used?** Ensure `torch.cuda.is_available()` returns `True` and the model is moved to `cuda`.

## License
This project is for educational purposes and follows OpenAI's CLIP model usage guidelines.

