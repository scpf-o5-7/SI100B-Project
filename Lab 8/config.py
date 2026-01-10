import torch
from typing import Dict, Any


class Config:

    HF_DATASET_NAME = "clip-benchmark/wds_fer2013"
    HF_DATASET_SPLIT_TRAIN = "train"
    HF_DATASET_SPLIT_VAL = "test"

    MODEL_SAVE_PATH = "models/model.pth"
    OUTPUT_DIR = "outputs"

    IMAGE_SIZE = 224
    NUM_CLASSES = 7
    CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    CONFIDENCE_THRESHOLD = 0.5

    FACE_DETECTION_MARGIN = 20
    MIN_FACE_SIZE = 30

    @staticmethod
    def to_dict() -> Dict[str, Any]:
        """转换为字典"""
        return {
            k: v
            for k, v in Config.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
