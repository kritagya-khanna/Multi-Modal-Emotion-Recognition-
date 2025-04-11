import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path


PROJECT_ROOT = Path("d:/multimodal emotion recognition system")
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_PATH = PROJECT_ROOT / "processed"
AUDIO_SPEECH_PATH = DATA_PATH / "Audio_Speech_Actors_01-24"
AUDIO_SONG_PATH = DATA_PATH / "Audio_Song_Actors_01-24"
VIDEO_PATH = DATA_PATH 

os.makedirs(PROCESSED_PATH / "audio_features", exist_ok=True)
os.makedirs(PROCESSED_PATH / "video_features", exist_ok=True)
os.makedirs(PROCESSED_PATH / "combined_features", exist_ok=True)

print("Project directories created successfully.")

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("Data setup complete.")