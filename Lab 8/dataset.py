"""
数据集处理
"""

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
    """基于Hugging Face数据集的人脸表情数据集"""

    def __init__(self, split: str = "train", is_train: bool = True):
        """
        初始化数据集
        
        Args:
            split: 数据集分割 ('train' 或 'test')
            is_train: 是否为训练模式
        """
        self.split = split
        self.is_train = is_train
        self.class_names = config.Config.CLASS_NAMES
        
        # 加载Hugging Face数据集
        self.dataset = datasets.load_dataset(
            config.Config.HF_DATASET_NAME,
            split=split
        )
        
        # 数据增强
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """获取数据增强转换"""
        if self.is_train:
            return A.Compose([
                A.Resize(config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE),  # 从48x48调整到224x224
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, 
                              min_holes=1, min_height=10, min_width=10, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # 弹性变形
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),  # 运动模糊
                    A.MedianBlur(blur_limit=3, p=0.5),  # 中值模糊
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE),  # 从48x48调整到224x224
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 获取样本
        sample = self.dataset[idx]
        
        # 提取图像 - 使用jpg字段
        if 'jpg' in sample:
            pil_image = sample['jpg']
            # 将PIL图像转换为numpy数组
            image = np.array(pil_image)
        else:
            raise KeyError("数据集中没有找到jpg图像字段")
        
        # 灰度图转换为RGB (48x48 灰度 -> 224x224 RGB)
        if len(image.shape) == 2:  # 灰度图 (H, W)
            image = np.stack([image] * 3, axis=-1)  # 转换为 (H, W, 3)
        
        # 获取标签 - 使用cls字段
        if 'cls' in sample:
            label = int(sample['cls'])
        else:
            raise KeyError("数据集中没有找到cls标签字段")
        
        # 应用转换
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class FaceDetector:
    """人脸检测器"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化人脸检测器

        Args:
            device: 计算设备
        """
        self.device = torch.device(device)
        # 使用MTCNN进行人脸检测
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
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (RGB格式)

        Returns:
            faces: 裁剪出的人脸图像列表
            boxes: 人脸边界框列表 (x1, y1, x2, y2)
        """
        # 转换为PIL图像
        image_pil = Image.fromarray(image)

        # 检测人脸
        boxes, probs = self.mtcnn.detect(image_pil)

        faces = []
        boxes_list = []

        if boxes is not None:
            h, w = image.shape[:2]
            for box, prob in zip(boxes, probs):
                if prob > config.Config.CONFIDENCE_THRESHOLD:
                    # 扩展边界框
                    x1, y1, x2, y2 = box
                    margin = config.Config.FACE_DETECTION_MARGIN
                    x1 = max(0, int(x1) - margin)
                    y1 = max(0, int(y1) - margin)
                    x2 = min(w, int(x2) + margin)
                    y2 = min(h, int(y2) + margin)

                    # 裁剪人脸
                    face = image[y1:y2, x1:x2]
                    if face.size > 0:
                        faces.append(face)
                        boxes_list.append((x1, y1, x2, y2))

        return faces, boxes_list


def get_data_loaders(batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    获取数据加载器（基于Hugging Face数据集）
    """
    if batch_size is None:
        batch_size = config.Config.BATCH_SIZE

    # 创建数据集
    train_dataset = FaceEmotionDataset(
        split=config.Config.HF_DATASET_SPLIT_TRAIN, is_train=True
    )
    val_dataset = FaceEmotionDataset(
        split=config.Config.HF_DATASET_SPLIT_VAL, is_train=False
    )

    # 创建数据加载器
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
