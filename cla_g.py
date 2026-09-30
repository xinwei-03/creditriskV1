#!/usr/bin/env python
# coding: utf-8

# 首先检查并安装必要的库
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 安装可能缺失的库
install_package("typing_extensions")
install_package("pandas")
install_package("numpy")
install_package("scikit-learn")
install_package("torch")
install_package("matplotlib")
install_package("networkx")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, \
    confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph

# Load data
input_csv = r"./test.csv"
df = pd.read_csv(input_csv)

# Preprocess categorical features
categorical_features = df.select_dtypes(include=["object",'bool']).columns
for feature in categorical_features:
    df[feature] = df[feature].astype('category').cat.codes

target = df['loan_status'].values
features = df.drop('loan_status', axis=1)

# 修正1: 先划分数据集，避免数据泄露
X_temp, X_test, y_temp, y_test = train_test_split(features.values, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 修正2: 只在训练集上进行特征选择和标准化
# 将数组转换回DataFrame以便相关性分析
X_train_df = pd.DataFrame(X_train, columns=features.columns)
X_val_df = pd.DataFrame(X_val, columns=features.columns)
X_test_df = pd.DataFrame(X_test, columns=features.columns)

# 在训练集上计算相关性
correlation_matrix = X_train_df.corr()
correlation_threshold = 0.6
highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

selected_features = [feature for feature in features.columns if feature not in highly_correlated_features]
print(f'Selected features: {len(selected_features)}')

# 应用特征选择到所有数据集
X_train_selected = X_train_df[selected_features].values
X_val_selected = X_val_df[selected_features].values
X_test_selected = X_test_df[selected_features].values

# ===============================
# 特征增强部分
# ===============================

def time_series_augmentation(X, method='difference', window_size=3):
    """
    时间层面特征增强
    方法1: 差分特征 - 计算滑动窗口内的统计量
    方法2: 滚动统计特征 - 均值、标准差等
    """
    n_samples, n_features = X.shape
    augmented_features = []
    
    if method == 'difference':
        # 添加差分特征 (当前值与前一个值的差)
        diff_features = np.zeros((n_samples, n_features))
        diff_features[1:] = X[1:] - X[:-1]
        augmented_features.append(diff_features)
        
    elif method == 'rolling_stats':
        # 滚动统计特征
        for window in [2, 3, 5]:
            if n_samples >= window:
                # 滚动均值
                rolling_mean = np.zeros((n_samples, n_features))
                for i in range(window-1, n_samples):
                    rolling_mean[i] = np.mean(X[i-window+1:i+1], axis=0)
                augmented_features.append(rolling_mean)
                
                # 滚动标准差
                rolling_std = np.zeros((n_samples, n_features))
                for i in range(window-1, n_samples):
                    rolling_std[i] = np.std(X[i-window+1:i+1], axis=0)
                augmented_features.append(rolling_std)
    
    # 将所有增强特征拼接
    if augmented_features:
        X_augmented = np.concatenate([X] + augmented_features, axis=1)
        return X_augmented
    else:
        return X

def gnn_spatial_augmentation(X, k=5, method='knn'):
    """
    空间GNN特征增强
    基于样本相似性构建图，并传播特征
    """
    n_samples, n_features = X.shape
    
    if method == 'knn':
        # 使用KNN构建图
        knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=True)
        knn_graph = knn_graph.toarray()
        
        # 图特征传播：每个节点的特征是其邻居特征的加权平均
        propagated_features = np.zeros_like(X)
        for i in range(n_samples):
            neighbors = np.where(knn_graph[i] > 0)[0]
            if len(neighbors) > 0:
                # 计算邻居特征的加权平均
                weights = knn_graph[i, neighbors]
                neighbor_features = X[neighbors]
                weighted_avg = np.average(neighbor_features, weights=weights, axis=0)
                propagated_features[i] = weighted_avg
        
        # 组合原始特征和图传播特征
        X_augmented = np.concatenate([X, propagated_features], axis=1)
        return X_augmented
    
    return X

# 应用特征增强（只在训练集上）
print("Applying feature augmentation...")

# 时间层面增强
X_train_time_aug = time_series_augmentation(X_train_selected, method='difference')
print(f"After time augmentation: {X_train_time_aug.shape}")

# 空间GNN增强
X_train_spatial_aug = gnn_spatial_augmentation(X_train_time_aug, k=5, method='knn')
print(f"After spatial augmentation: {X_train_spatial_aug.shape}")

# 对验证集和测试集也应用相同的增强（使用训练集的模式）
X_val_time_aug = time_series_augmentation(X_val_selected, method='difference')
X_val_spatial_aug = gnn_spatial_augmentation(X_val_time_aug, k=5, method='knn')

X_test_time_aug = time_series_augmentation(X_test_selected, method='difference') 
X_test_spatial_aug = gnn_spatial_augmentation(X_test_time_aug, k=5, method='knn')

# 更新特征集
X_train_augmented = X_train_spatial_aug
X_val_augmented = X_val_spatial_aug
X_test_augmented = X_test_spatial_aug

print(f'Final augmented feature shapes - Train: {X_train_augmented.shape}, Val: {X_val_augmented.shape}, Test: {X_test_augmented.shape}')

# 修正3: 只在训练集上拟合标准化器（使用增强后的特征）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_augmented)
X_val_scaled = scaler.transform(X_val_augmented)
X_test_scaled = scaler.transform(X_test_augmented)

print(f'Scaled feature shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}')

# Convert to PyTorch tensors and DataLoader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Define CNN-LSTM-Attention model (更新输入维度)
class CNNLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=64, lstm_hidden_dim=64, num_lstm_layers=1, num_classes=1, dropout=0.3):
        super(CNNLSTMAttentionClassifier, self).__init__()
        
        # CNN部分 - 用于提取局部特征
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM部分 - 用于捕捉序列依赖关系
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.Tanh(),
            nn.Linear(lstm_hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类层
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, 64)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len)
        
        # CNN处理
        cnn_out = self.conv1d_1(x)
        cnn_out = self.relu1(cnn_out)
        cnn_out = self.bn1(cnn_out)
        
        cnn_out = self.conv1d_2(cnn_out)
        cnn_out = self.relu2(cnn_out)
        cnn_out = self.bn2(cnn_out)
        cnn_out = self.dropout1(cnn_out)
        
        # 调整维度用于LSTM: (batch_size, seq_len, features)
        lstm_input = cnn_out.transpose(1, 2)
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # 应用注意力权重
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector shape: (batch_size, hidden_dim * 2)
        
        # 分类
        out = self.fc1(context_vector)
        out = self.relu3(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def train_cnn_lstm_attention_classifier(model, dataloader, val_dataloader, optimizer, class_weights, epochs=20):
    model.train()
    train_losses = []
    val_losses = []
    criterion = nn.BCELoss(reduction='none')  # 使用reduction='none'以便手动应用权重

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            pred_y = model(data)
            weights = class_weights[target.long().squeeze()]  # 动态计算权重
            loss = criterion(pred_y, target)
            loss = (loss * weights).mean()  # 应用权重并取平均值
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_losses.append(train_loss / len(dataloader.dataset))
        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.6f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_dataloader:
                pred_y = model(data)
                weights = class_weights[target.long().squeeze()]  # 动态计算权重
                loss = criterion(pred_y, target)
                loss = (loss * weights).mean()  # 应用权重并取平均值
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataloader.dataset))
        print(f'Epoch {epoch + 1}, Val Loss: {val_losses[-1]:.6f}')

    return train_losses, val_losses

# Parameters for model (更新输入维度)
input_dim = X_train_scaled.shape[1]  # 使用增强后的特征维度
cnn_out_channels = 64
lstm_hidden_dim = 64
num_lstm_layers = 2
output_dim = 1  # Binary classification

# Initialize and train the CNNLSTMAttentionClassifier
cnn_lstm_attention_classifier = CNNLSTMAttentionClassifier(
    input_dim=input_dim,
    cnn_out_channels=cnn_out_channels,
    lstm_hidden_dim=lstm_hidden_dim,
    num_lstm_layers=num_lstm_layers,
    num_classes=output_dim
)
optimizer = optim.Adam(cnn_lstm_attention_classifier.parameters(), lr=1e-3, weight_decay=1e-5)

print("Starting training with augmented features...")
train_losses, val_losses = train_cnn_lstm_attention_classifier(
    cnn_lstm_attention_classifier, train_dataloader, val_dataloader, optimizer, class_weights
)

# Predict and evaluate on the test set
cnn_lstm_attention_classifier.eval()
with torch.no_grad():
    pred_y_test = cnn_lstm_attention_classifier(X_test_tensor).numpy().squeeze()

# Evaluate the model
y_pred = (pred_y_test > 0.5).astype(int)
roc_auc = roc_auc_score(y_test, pred_y_test)  
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
gini = 2 * roc_auc - 1

# 计算 KS 统计量
def ks_statistic(y_true, pred_y_test):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': pred_y_test})
    data = data.sort_values(by='y_prob', ascending=False)
    data['cumulative_good'] = (data['y_true'] == 0).cumsum() / (data['y_true'] == 0).sum()
    data['cumulative_bad'] = (data['y_true'] == 1).cumsum() / (data['y_true'] == 1).sum()
    data['ks'] = np.abs(data['cumulative_bad'] - data['cumulative_good'])
    ks_stat = data['ks'].max()
    return ks_stat

ks_stat = ks_statistic(y_test, pred_y_test)

print(f'CNN-LSTM-Attention with Feature Augmentation: ##############')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Gini Coefficient: {gini:.4f}')
print(f'KS Statistic: {ks_stat:.4f}')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - CNN-LSTM-Attention with Feature Augmentation')
plt.legend()
plt.grid(True)
plt.show()

# Calculate confusion matrix and Type II Error
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
type_II_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type II Error (False Negative Rate): {type_II_error:.4f}')
print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')