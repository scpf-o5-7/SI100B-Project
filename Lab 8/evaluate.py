import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, recall_score, precision_score, f1_score)
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
from dataset import get_data_loaders
from models import EfficientNetEmotionClassifier
from utils import plot_confusion_matrix, plot_metrics_comparison

def evaluate_model(model_path: str = None, save_plots: bool = True):
    """评估模型"""
    cfg = config.Config
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(batch_size=32)

    print(f"Using dataset: {cfg.HF_DATASET_NAME}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 加载模型
    if model_path is None:
        model_path = cfg.MODEL_SAVE_PATH
    
    model = EfficientNetEmotionClassifier(
        num_classes=cfg.NUM_CLASSES,
        pretrained=False
    ).to(cfg.DEVICE)
    
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 评估函数
    def evaluate_loader(data_loader, loader_name: str):
        """评估数据加载器"""
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc=f'Evaluating {loader_name}'):
                inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        return all_preds, all_targets, all_probs
    
    # 评估训练集和验证集
    print("Evaluating training set...")
    train_preds, train_targets, train_probs = evaluate_loader(train_loader, "training")
    
    print("\nEvaluating validation set...")
    val_preds, val_targets, val_probs = evaluate_loader(val_loader, "validation")
    
    # 计算指标
    def calculate_metrics(preds, targets, probs, dataset_name: str):
        """计算评估指标"""
        # 自动检测实际存在的类别
        unique_classes = np.unique(targets)
        actual_num_classes = len(unique_classes)
        actual_class_names = [cfg.CLASS_NAMES[i] for i in unique_classes]
        
        print(f"检测到实际类别数: {actual_num_classes}")
        print(f"实际类别: {actual_class_names}")
        
        # 基础指标
        accuracy = accuracy_score(targets, preds)
        
        # 多类别召回率（每类单独计算，然后取平均）
        recall_per_class = recall_score(targets, preds, average=None, labels=unique_classes)
        recall_macro = recall_score(targets, preds, average='macro', labels=unique_classes)
        recall_weighted = recall_score(targets, preds, average='weighted', labels=unique_classes)
        
        # 精确率
        precision_macro = precision_score(targets, preds, average='macro', labels=unique_classes)
        precision_weighted = precision_score(targets, preds, average='weighted', labels=unique_classes)
        
        # F1分数
        f1_macro = f1_score(targets, preds, average='macro', labels=unique_classes)
        f1_weighted = f1_score(targets, preds, average='weighted', labels=unique_classes)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, preds, labels=unique_classes, normalize='true')
        
        # 分类报告
        report = classification_report(
            targets, 
            preds, 
            target_names=actual_class_names,
            labels=unique_classes,
            output_dict=True
        )
        
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'recall_per_class': recall_per_class,
            'confusion_matrix': cm,
            'classification_report': report,
            'actual_num_classes': actual_num_classes,
            'actual_class_names': actual_class_names,
            'unique_classes': unique_classes
        }
        
        return metrics
    
    # 计算训练集和验证集指标
    train_metrics = calculate_metrics(train_preds, train_targets, train_probs, "Training")
    val_metrics = calculate_metrics(val_preds, val_targets, val_probs, "Validation")
    
    # 打印结果
    def print_metrics(metrics):
        """打印指标"""
        print(f"\n{'='*50}")
        print(f"{metrics['dataset']} Set Metrics:")
        print(f"{'='*50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Recall: {metrics['recall_macro']:.4f}")
        print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        print(f"Macro Precision: {metrics['precision_macro']:.4f}")
        print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-class Recall:")
        # 使用实际检测到的类别，而不是配置文件中的所有类别
        for i, class_name in enumerate(metrics['actual_class_names']):
            print(f"  {class_name}: {metrics['recall_per_class'][i]:.4f}")
    
    print_metrics(train_metrics)
    print_metrics(val_metrics)
    
    # 创建输出目录
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if save_plots:
        # 绘制混淆矩阵
        print("\nGenerating confusion matrices...")
        
        # 训练集混淆矩阵
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(
            train_metrics['confusion_matrix'],
            classes=train_metrics['actual_class_names'],  # 使用实际类别
            title=f'Confusion Matrix - Training Set (Acc: {train_metrics["accuracy"]:.3f})',
            normalize=True
        )
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_train.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证集混淆矩阵
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(
            val_metrics['confusion_matrix'],
            classes=val_metrics['actual_class_names'],  # 使用实际类别
            title=f'Confusion Matrix - Validation Set (Acc: {val_metrics["accuracy"]:.3f})',
            normalize=True
        )
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_val.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制指标对比图
        plot_metrics_comparison(train_metrics, val_metrics, output_dir)
        
        # 保存详细结果到CSV
        save_detailed_results(train_metrics, val_metrics, output_dir)
        
        print(f"\nResults saved to: {output_dir}/")
    
    return train_metrics, val_metrics


def save_detailed_results(train_metrics, val_metrics, output_dir: str):
    """保存详细结果到CSV"""
    # 创建数据框
    results = []
    
    # 使用实际检测到的类别
    for i, class_name in enumerate(train_metrics['actual_class_names']):
        train_recall = train_metrics['recall_per_class'][i]
        
        # 在验证集中查找相同类别的索引
        if class_name in val_metrics['actual_class_names']:
            val_index = val_metrics['actual_class_names'].index(class_name)
            val_recall = val_metrics['recall_per_class'][val_index]
        else:
            val_recall = 0.0  # 如果验证集中没有这个类别
        
        results.append({
            'Class': class_name,
            'Train_Recall': train_recall,
            'Val_Recall': val_recall,
            'Recall_Diff': val_recall - train_recall
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'per_class_recall.csv'), index=False)
    
    # 保存总体指标
    overall_metrics = pd.DataFrame([
        {
            'Dataset': 'Training',
            'Accuracy': train_metrics['accuracy'],
            'Macro_Recall': train_metrics['recall_macro'],
            'Weighted_Recall': train_metrics['recall_weighted'],
            'Macro_Precision': train_metrics['precision_macro'],
            'Weighted_Precision': train_metrics['precision_weighted'],
            'Macro_F1': train_metrics['f1_macro'],
            'Weighted_F1': train_metrics['f1_weighted']
        },
        {
            'Dataset': 'Validation',
            'Accuracy': val_metrics['accuracy'],
            'Macro_Recall': val_metrics['recall_macro'],
            'Weighted_Recall': val_metrics['recall_weighted'],
            'Macro_Precision': val_metrics['precision_macro'],
            'Weighted_Precision': val_metrics['precision_weighted'],
            'Macro_F1': val_metrics['f1_macro'],
            'Weighted_F1': val_metrics['f1_weighted']
        }
    ])
    
    overall_metrics.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)
    
    print("Detailed results saved to CSV files.")


if __name__ == '__main__':
    # 评估模型
    train_metrics, val_metrics = evaluate_model()
    
    # 打印总结
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Training Accuracy:   {train_metrics['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Training Macro Recall:   {train_metrics['recall_macro']:.4f}")
    print(f"Validation Macro Recall: {val_metrics['recall_macro']:.4f}")
    print("="*60)