import datasets
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from typing import Tuple, List
import config


class FaceEmotionDataset(Dataset):

    def __init__(self, split: str = "train", is_train: bool = True):
        self.split = split
        self.is_train = is_train
        self.class_names = config.Config.CLASS_NAMES

        self.dataset = datasets.load_dataset(config.Config.HF_DATASET_NAME, split=split)

        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.1),
                        ],
                        p=0.3,
                    ),
                    A.Rotate(limit=15, p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.5),
                            A.HueSaturationValue(p=0.5),
                            A.CLAHE(p=0.3),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                            A.ISONoise(p=0.5),
                        ],
                        p=0.3,
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=20,
                        max_width=20,
                        min_holes=1,
                        min_height=10,
                        min_width=10,
                        p=0.3,
                    ),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=3, p=0.5),
                            A.MedianBlur(blur_limit=3, p=0.5),
                            A.GaussianBlur(blur_limit=3, p=0.3),
                        ],
                        p=0.3,
                    ),
                    A.RandomGamma(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        sample = self.dataset[idx]

        if "jpg" in sample:
            pil_image = sample["jpg"]

            image = np.array(pil_image)
        else:
            raise KeyError("No JPG image field was found in the dataset.")

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        if "cls" in sample:
            label = int(sample["cls"])
        else:
            raise KeyError("No cls label field was found in the dataset.")

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class FaceDetector:

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.device = torch.device(device)

        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device=self.device,
            min_face_size=config.Config.MIN_FACE_SIZE,
        )

    def detect_faces(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

        image_pil = Image.fromarray(image)

        boxes, probs = self.mtcnn.detect(image_pil)

        faces = []
        boxes_list = []

        if boxes is not None:
            h, w = image.shape[:2]
            for box, prob in zip(boxes, probs):
                if prob > config.Config.CONFIDENCE_THRESHOLD:

                    x1, y1, x2, y2 = box
                    margin = config.Config.FACE_DETECTION_MARGIN
                    x1 = max(0, int(x1) - margin)
                    y1 = max(0, int(y1) - margin)
                    x2 = min(w, int(x2) + margin)
                    y2 = min(h, int(y2) + margin)

                    face = image[y1:y2, x1:x2]
                    if face.size > 0:
                        faces.append(face)
                        boxes_list.append((x1, y1, x2, y2))

        return faces, boxes_list


def get_data_loaders(batch_size: int = None) -> Tuple[DataLoader, DataLoader]:

    if batch_size is None:
        batch_size = config.Config.BATCH_SIZE

    train_dataset = FaceEmotionDataset(
        split=config.Config.HF_DATASET_SPLIT_TRAIN, is_train=True
    )
    val_dataset = FaceEmotionDataset(
        split=config.Config.HF_DATASET_SPLIT_VAL, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
