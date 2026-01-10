import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
from typing import Tuple, List, Optional
import config
import os
import dataset
import cv2


class EfficientNetEmotionClassifier(nn.Module):

    def __init__(self, num_classes: int = None, pretrained: bool = True):

        super().__init__()

        if num_classes is None:
            num_classes = config.Config.NUM_CLASSES

        self.backbone = timm.create_model(
            config.Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            features_only=False,
        )

        with torch.no_grad():
            dummy_input = torch.randn(
                1, 3, config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE
            )
            features = self.backbone(dummy_input)
            if features.dim() == 4:
                features = nn.AdaptiveAvgPool2d(1)(features)
                features = features.view(features.size(0), -1)
            in_features = features.shape[1]

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.Config.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.Config.DROPOUT_RATE / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.backbone(x)

        if features.dim() == 4:

            se_weights = self.se_block(features)
            se_weights = se_weights.unsqueeze(-1).unsqueeze(-1)
            weighted_features = features * se_weights

            pooled_features = nn.AdaptiveAvgPool2d(1)(weighted_features)
            flattened_features = pooled_features.view(pooled_features.size(0), -1)
        else:

            if features.dim() == 2:

                spatial_features = features.unsqueeze(-1).unsqueeze(-1)
                se_weights = self.se_block(spatial_features)
                flattened_features = features * se_weights
            else:
                flattened_features = features.view(features.size(0), -1)

        logits = self.classifier(flattened_features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        if features.dim() == 4:
            features = nn.AdaptiveAvgPool2d(1)(features)
            features = features.view(features.size(0), -1)

        return features


class FaceEmotionSystem:

    def __init__(self, model_path: Optional[str] = None):

        self.device = config.Config.DEVICE
        self.class_names = config.Config.CLASS_NAMES

        self.face_detector = dataset.FaceDetector(str(self.device))
        self.model = self._load_model(model_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        model = EfficientNetEmotionClassifier(
            num_classes=config.Config.NUM_CLASSES, pretrained=False
        )

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Using randomly initialized model")

        model = model.to(self.device)
        model.eval()
        return model

    def detect_and_classify(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces, boxes = self.face_detector.detect_faces(image_rgb)

        emotions = []
        confidences = []

        if faces:

            processed_faces = []
            for face in faces:

                face_resized = cv2.resize(
                    face, (config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE)
                )

                face_tensor = self.transform(face_resized)
                processed_faces.append(face_tensor)

            with torch.no_grad():
                batch = torch.stack(processed_faces).to(self.device)
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

                emotions = [self.class_names[pred] for pred in preds.cpu().numpy()]
                confidences = confs.cpu().numpy().tolist()

        return boxes, emotions, confidences

    def draw_results(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        emotions: List[str],
        confidences: List[float],
    ) -> np.ndarray:

        result = image.copy()

        for (x1, y1, x2, y2), emotion, confidence in zip(boxes, emotions, confidences):

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{emotion}: {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return result
