"""
模型定义
"""
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
    """基于EfficientNet的表情分类器"""
    
    def __init__(self, num_classes: int = None, pretrained: bool = True):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
        """
        super().__init__()
        
        if num_classes is None:
            num_classes = config.Config.NUM_CLASSES
            
        # 使用timm库中的EfficientNet
        self.backbone = timm.create_model(
            config.Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,  # 不使用分类头
            features_only=False
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE)
            features = self.backbone(dummy_input)  # 获取backbone输出
            # 调整特征形状：如果输出是[B, C, H, W]，则全局平均池化后为[B, C]
            if features.dim() == 4:  # 如果是4D张量，需要池化
                features = nn.AdaptiveAvgPool2d(1)(features)  # 全局平均池化
                features = features.view(features.size(0), -1)  # 展平
            in_features = features.shape[1]  # 现在in_features已定义

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_features, in_features // 16),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features),  # 恢复通道
            nn.Sigmoid()  # 生成通道权重
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.Config.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.Config.DROPOUT_RATE / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # 初始化分类头
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 3, H, W]
            
        Returns:
            logits: 输出logits
        """
        features = self.backbone(x)

        # 新增SE块处理
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)
        se_weights = self.se_block(features)  # 生成通道权重[B, C]
        se_weights = se_weights.unsqueeze(-1).unsqueeze(-1)  # 调整为[B, C, 1, 1]
        weighted_features = features * se_weights  # 通道重加权

        if weighted_features.dim() == 4:
            weighted_features = weighted_features.view(weighted_features.size(0), -1)
        logits = self.classifier(weighted_features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征向量"""
        return self.backbone(x)


class FaceEmotionSystem:
    """完整的人脸表情识别系统"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化系统
        
        Args:
            model_path: 模型路径
        """
        self.device = config.Config.DEVICE
        self.class_names = config.Config.CLASS_NAMES
        
        # 初始化组件
        self.face_detector = dataset.FaceDetector(str(self.device))
        self.model = self._load_model(model_path)
        
        # 预处理转换
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """加载模型"""
        model = EfficientNetEmotionClassifier(
            num_classes=config.Config.NUM_CLASSES,
            pretrained=False
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Using randomly initialized model")
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def detect_and_classify(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], 
                                                             List[str], List[float]]:
        """
        检测人脸并进行表情分类
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 人脸边界框列表
            emotions: 表情标签列表
            confidences: 置信度列表
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        faces, boxes = self.face_detector.detect_faces(image_rgb)
        
        emotions = []
        confidences = []
        
        if faces:
            # 预处理所有人脸
            processed_faces = []
            for face in faces:
                # 调整大小
                face_resized = cv2.resize(face, (config.Config.IMAGE_SIZE, config.Config.IMAGE_SIZE))
                # 转换
                face_tensor = self.transform(face_resized)
                processed_faces.append(face_tensor)
            
            # 批量预测
            with torch.no_grad():
                batch = torch.stack(processed_faces).to(self.device)
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                
                emotions = [self.class_names[pred] for pred in preds.cpu().numpy()]
                confidences = confs.cpu().numpy().tolist()
        
        return boxes, emotions, confidences
    
    def draw_results(self, image: np.ndarray, 
                     boxes: List[Tuple[int, int, int, int]], 
                     emotions: List[str], 
                     confidences: List[float]) -> np.ndarray:
        """
        在图像上绘制结果
        
        Args:
            image: 原始图像
            boxes: 边界框列表
            emotions: 表情列表
            confidences: 置信度列表
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        for (x1, y1, x2, y2), emotion, confidence in zip(boxes, emotions, confidences):
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            label = f"{emotion}: {confidence:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(result, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(result, label,
                       (x1, y1 - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result
    
def get_features(self, x):
    return self.backbone(x)