#!/usr/bin/env python3
"""
DeepFM PyTorch 模型训练脚本

用法:
    python train/train_deepfm.py [--version VERSION] [--epochs EPOCHS] [--batch-size BATCH_SIZE]

功能:
    1. 读取训练数据（与 train_xgb.py 格式一致）
    2. 切分训练集/验证集
    3. 训练 DeepFM 模型（PyTorch）
    4. 保存模型和特征元数据（支持版本管理）

DeepFM 模型结构:
    - FM 部分：二阶特征交互（Factorization Machine）
    - Deep 部分：DNN 处理高阶非线性交互
    - 输出：FM + Deep 联合预测
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入特征配置
import importlib.util
features_path = os.path.join(os.path.dirname(__file__), "features.py")
spec = importlib.util.spec_from_file_location("features", features_path)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)
FEATURE_COLUMNS = features.FEATURE_COLUMNS
FEATURE_META_PATH = features.FEATURE_META_PATH
LABEL_COLUMN = features.LABEL_COLUMN
MODEL_DIR = features.MODEL_DIR

# DeepFM 模型路径
DEEPFM_MODEL_PATH = os.path.join(MODEL_DIR, "deepfm_model.pt")


class DeepFMDataset(Dataset):
    """DeepFM 数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepFM(nn.Module):
    """
    DeepFM 模型（简化版：所有特征视为稠密特征）
    
    结构:
        - FM 部分：二阶特征交互（Factorization Machine）
        - Deep 部分：多层 DNN
        - 输出：FM + Deep + Bias
    
    注意：本实现将所有特征视为稠密特征（连续值），使用线性变换生成 embedding。
    实际应用中，应该区分稀疏特征（categorical）和稠密特征（numerical），
    稀疏特征使用 Embedding，稠密特征使用线性变换或直接输入。
    """
    
    def __init__(self, num_features, embedding_dim=16, deep_layers=[128, 64, 32], dropout=0.5):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # FM 部分：每个特征一个线性变换生成 embedding（适用于稠密特征）
        # 对于稠密特征，使用线性变换：embedding = W * feature + b
        self.fm_embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        
        # FM 线性层（一阶项）
        self.linear = nn.Linear(num_features, 1)
        
        # Deep 部分：输入为原始特征（稠密特征直接输入）
        deep_input_dim = num_features
        deep_layers = [deep_input_dim] + deep_layers
        self.deep = nn.ModuleList()
        for i in range(len(deep_layers) - 1):
            self.deep.append(nn.Linear(deep_layers[i], deep_layers[i + 1]))
            self.deep.append(nn.ReLU())
            self.deep.append(nn.Dropout(dropout))
        
        # Deep 输出层
        self.deep_output = nn.Linear(deep_layers[-1], 1)
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, num_features] 特征矩阵（已标准化）
        
        Returns:
            [batch_size, 1] 预测分数
        """
        batch_size = x.size(0)
        
        # FM 部分：一阶项
        linear_part = self.linear(x)  # [batch_size, 1]
        
        # FM 部分：二阶项（通过线性变换生成 embedding）
        embeddings_list = []
        for i in range(self.num_features):
            # 对每个特征进行线性变换生成 embedding
            feat = x[:, i:i+1]  # [batch_size, 1]
            emb = self.fm_embeddings[i](feat)  # [batch_size, embedding_dim]
            embeddings_list.append(emb)
        
        # 计算二阶交互：sum_square - square_sum
        embeddings = torch.stack(embeddings_list, dim=1)  # [batch_size, num_features, embedding_dim]
        sum_square = torch.sum(embeddings, dim=1) ** 2  # [batch_size, embedding_dim]
        square_sum = torch.sum(embeddings ** 2, dim=1)  # [batch_size, embedding_dim]
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # [batch_size, 1]
        
        # Deep 部分：直接使用原始特征
        deep_output = x
        for layer in self.deep:
            deep_output = layer(deep_output)
        deep_output = self.deep_output(deep_output)  # [batch_size, 1]
        
        # 联合输出
        output = linear_part + fm_second_order + deep_output + self.bias
        return torch.sigmoid(output)  # 二分类，输出 0-1 之间


def generate_sample_data(output_path: str, n_samples: int = 1000):
    """生成示例训练数据（与 train_xgb.py 一致）"""
    np.random.seed(42)
    
    item_ctr = np.random.uniform(0.01, 0.5, n_samples)
    item_cvr = np.random.uniform(0.001, 0.1, n_samples)
    item_price = np.random.uniform(10, 200, n_samples)
    user_age = np.random.randint(18, 60, n_samples).astype(np.float64)
    user_gender = np.random.randint(0, 3, n_samples).astype(np.float64)
    
    data = {
        "item_ctr": item_ctr,
        "item_cvr": item_cvr,
        "item_price": item_price,
        "user_age": user_age,
        "user_gender": user_gender,
        "cross_age_x_ctr": user_age * item_ctr,
        "cross_gender_x_price": user_gender * item_price,
    }
    
    label = (
        0.5 * data["item_ctr"] * 10
        + 0.3 * data["item_cvr"] * 20
        + 0.1 * data["user_age"] / 100
        + 0.05 * data["user_gender"] / 2.0
        + np.random.normal(0, 0.1, n_samples)
    )
    data[LABEL_COLUMN] = (label > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"生成示例数据: {output_path}, 样本数: {n_samples}")
    return df


def train_model(data_path: str, model_version: str = None, epochs: int = 50, batch_size: int = 32):
    """
    训练 DeepFM 模型
    
    Args:
        data_path: 训练数据路径
        model_version: 模型版本（可选）
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print(f"读取数据: {data_path}")
    
    # 读取数据
    if not os.path.exists(data_path):
        print(f"数据文件不存在，生成示例数据...")
        df = generate_sample_data(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"特征列: {FEATURE_COLUMNS}")
    
    # 检查必要的列
    missing_cols = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 准备特征和标签
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.float32)
    
    # 特征标准化（DeepFM 需要）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("特征标准化完成")
    
    # 切分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    # 创建数据集和数据加载器
    train_dataset = DeepFMDataset(X_train, y_train)
    val_dataset = DeepFMDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    num_features = len(FEATURE_COLUMNS)
    model = DeepFM(
        num_features=num_features,
        embedding_dim=16,
        deep_layers=[128, 64, 32],
        dropout=0.5
    )
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n使用设备: {device}")
    print(f"开始训练 DeepFM 模型...")
    print(f"模型参数: embedding_dim=16, deep_layers=[128, 64, 32], dropout=0.5")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr=0.001")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 Epoch {epoch+1}")
                break
    
    # 生成模型版本
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_features': num_features,
            'embedding_dim': 16,
            'deep_layers': [128, 64, 32],
            'dropout': 0.5,
        },
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'model_version': model_version,
    }, DEEPFM_MODEL_PATH)
    print(f"\n模型已保存: {DEEPFM_MODEL_PATH}")
    
    # 保存特征元数据
    feature_meta = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "model_version": model_version,
        "normalized": True,
        "model_type": "deepfm",
        "created_at": datetime.now().isoformat(),
    }
    with open(FEATURE_META_PATH.replace("feature_meta.json", "deepfm_feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"特征元数据已保存: {FEATURE_META_PATH.replace('feature_meta.json', 'deepfm_feature_meta.json')}")
    
    return model, feature_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 DeepFM 模型")
    parser.add_argument("--version", type=str, help="模型版本（可选，默认使用时间戳）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50）")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小（默认 32）")
    args = parser.parse_args()
    
    # 数据路径
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "train_data.csv")
    
    try:
        model, feature_meta = train_model(
            data_path,
            model_version=args.version,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        print("\n训练完成！")
        print(f"模型版本: {feature_meta['model_version']}")
        print(f"模型文件: {DEEPFM_MODEL_PATH}")
        print(f"特征元数据: {FEATURE_META_PATH.replace('feature_meta.json', 'deepfm_feature_meta.json')}")
        print("\n下一步: 启动推理服务")
        print("  cd python")
        print("  uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080")
    except Exception as e:
        print(f"训练失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
