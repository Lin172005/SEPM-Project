import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DATABASE = {
    'name': 'anomaly_detection.db',
    'path': str(BASE_DIR / 'anomaly_detection.db')
}

# File storage paths
STORAGE = {
    'incidents': str(BASE_DIR / 'incidents'),
    'recordings': str(BASE_DIR / 'recordings'),
    'logs': str(BASE_DIR / 'logs')
}

# Model configuration
MODEL = {
    'yolo_model': 'yolov8n.pt',
    'confidence_threshold': 0.4,
    'iou_threshold': 0.45
}

# Detection thresholds
THRESHOLDS = {
    'violence': 0.75,
    'min_violence_frames': 2,
    'proximity': 0.8,
    'movement': 0.15,
    'min_people': 2,
    'max_people': 4
}

# Email configuration
EMAIL = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'app_password': '',  # To be set by user
    'notification_enabled': True
}

# Camera configuration
CAMERA = {
    'default_width': 640,
    'default_height': 480,
    'default_fps': 30,
    'buffer_size': 1
}

# Create required directories
for directory in STORAGE.values():
    os.makedirs(directory, exist_ok=True) 