"""
工具函数
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional
import cv2
import config

def save_checkpoint(state: Dict, filename: str = 'checkpoint.pth'):
    """
    保存检查点
    
    Args:
        state: 状态字典
        filename: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename: str, model: torch.nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None):
    """
    加载检查点
    
    Args:
        filename: 检查点路径
        model: 模型
        optimizer: 优化器
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=config.Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from {filename}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)
    else:
        print(f"No checkpoint found at {filename}")
        return 0, 0.0


class EarlyStopping:
    """早停类"""
    
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        """
        Args:
            patience: 耐心值
            verbose: 是否打印信息
            delta: 最小改善阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, val_loss: float):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                          title: str = 'Confusion Matrix', 
                          normalize: bool = False,
                          cmap: plt.cm = plt.cm.Blues):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        classes: 类别名称
        title: 标题
        normalize: 是否归一化
        cmap: 颜色映射
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # 在格子中显示数值
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()


def plot_training_history(history: Dict):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = config.Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(train_metrics: Dict, val_metrics: Dict, output_dir: str):
    """
    绘制训练集和验证集指标对比
    
    Args:
        train_metrics: 训练集指标
        val_metrics: 验证集指标
        output_dir: 输出目录
    """
    metrics_names = ['Accuracy', 'Macro Recall', 'Weighted Recall', 
                    'Macro Precision', 'Weighted Precision',
                    'Macro F1', 'Weighted F1']
    
    train_values = [
        train_metrics['accuracy'],
        train_metrics['recall_macro'],
        train_metrics['recall_weighted'],
        train_metrics['precision_macro'],
        train_metrics['precision_weighted'],
        train_metrics['f1_macro'],
        train_metrics['f1_weighted']
    ]
    
    val_values = [
        val_metrics['accuracy'],
        val_metrics['recall_macro'],
        val_metrics['recall_weighted'],
        val_metrics['precision_macro'],
        val_metrics['precision_weighted'],
        val_metrics['f1_macro'],
        val_metrics['f1_weighted']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Training vs Validation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_results_grid(original_image: np.ndarray, 
                         boxes: List[Tuple[int, int, int, int]], 
                         emotions: List[str], 
                         confidences: List[float]) -> np.ndarray:
    """
    创建结果可视化网格
    
    Args:
        original_image: 原始图像
        boxes: 边界框列表
        emotions: 表情列表
        confidences: 置信度列表
        
    Returns:
        网格图像
    """
    if not boxes:
        return original_image
    
    # 创建副本
    result = original_image.copy()
    h, w = result.shape[:2]
    
    # 绘制边界框和标签
    for (x1, y1, x2, y2), emotion, confidence in zip(boxes, emotions, confidences):
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 标签文本
        label = f"{emotion}: {confidence:.2f}"
        
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # 绘制标签背景
        cv2.rectangle(result, 
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     (0, 255, 0), -1)
        
        # 绘制文本
        cv2.putText(result, label,
                   (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 如果有检测到人脸，创建人脸裁剪网格
    if boxes:
        # 创建子图网格
        n_faces = len(boxes)
        n_cols = min(3, n_faces)
        n_rows = (n_faces + n_cols - 1) // n_cols
        
        # 计算网格大小
        face_size = 100
        grid_h = n_rows * face_size
        grid_w = n_cols * face_size
        
        # 创建网格图像
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, ((x1, y1, x2, y2), emotion, confidence) in enumerate(zip(boxes, emotions, confidences)):
            row = i // n_cols
            col = i % n_cols
            
            # 裁剪人脸
            face = original_image[y1:y2, x1:x2]
            if face.size > 0:
                # 调整大小
                face_resized = cv2.resize(face, (face_size, face_size))
                
                # 添加到网格
                y_start = row * face_size
                y_end = (row + 1) * face_size
                x_start = col * face_size
                x_end = (col + 1) * face_size
                grid[y_start:y_end, x_start:x_end] = face_resized
                
                # 在网格中绘制标签
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(grid, label,
                          (x_start + 5, y_start + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 调整网格大小以匹配原始图像宽度
        grid_resized = cv2.resize(grid, (w, grid_h))
        
        # 合并原始图像和网格
        final = np.vstack([result, grid_resized])
        
        # 添加分隔线
        cv2.line(final, (0, h), (w, h), (255, 255, 255), 2)
        
        return final
    
    return result