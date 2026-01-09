import torch
import torch.nn as nn
import torch.nn.functional as F

class AdCorreLoss(nn.Module):
    def __init__(self, lambda_val=0.5, num_classes=7):
        super().__init__()
        self.lambda_val = lambda_val
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, features):
        # 交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)
        
        # Ad-Corre组件（特征判别器FD）：鼓励类内高相关、类间低相关
        batch_size = features.size(0)
        features_norm = F.normalize(features, p=2, dim=1)  # L2归一化
        correlation_matrix = torch.mm(features_norm, features_norm.t())  # [B, B]相关矩阵
        
        # 构建目标相关矩阵：类内为1，类间为-1
        target_matrix = torch.ones(batch_size, batch_size).to(features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if targets[i] == targets[j]:
                    target_matrix[i, j] = 1.0
                else:
                    target_matrix[i, j] = -1.0
        
        # 计算FD损失（均方误差）
        fd_loss = F.mse_loss(correlation_matrix, target_matrix)
        
        # 总损失
        total_loss = ce_loss + self.lambda_val * fd_loss
        return total_loss