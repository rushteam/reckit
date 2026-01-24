"""
DeepFM 模型加载器：封装 PyTorch DeepFM 模型的加载和预测逻辑
"""
import json
import logging
import os
import warnings

import numpy as np
import torch
import torch.nn as nn

# 配置日志
logger = logging.getLogger(__name__)

# 导入 DeepFM 模型定义（从训练脚本）
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# DeepFM 模型定义（与训练脚本一致）
class DeepFM(nn.Module):
    """DeepFM 模型（与训练脚本一致：所有特征视为稠密特征）"""
    
    def __init__(self, num_features, embedding_dim=16, deep_layers=[128, 64, 32], dropout=0.5):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # FM 部分：每个特征一个线性变换生成 embedding
        self.fm_embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        
        self.linear = nn.Linear(num_features, 1)
        
        # Deep 部分：输入为原始特征
        deep_input_dim = num_features
        deep_layers = [deep_input_dim] + deep_layers
        self.deep = nn.ModuleList()
        for i in range(len(deep_layers) - 1):
            self.deep.append(nn.Linear(deep_layers[i], deep_layers[i + 1]))
            self.deep.append(nn.ReLU())
            self.deep.append(nn.Dropout(dropout))
        
        self.deep_output = nn.Linear(deep_layers[-1], 1)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        linear_part = self.linear(x)
        
        embeddings_list = []
        for i in range(self.num_features):
            feat = x[:, i:i+1]
            emb = self.fm_embeddings[i](feat)
            embeddings_list.append(emb)
        
        embeddings = torch.stack(embeddings_list, dim=1)
        sum_square = torch.sum(embeddings, dim=1) ** 2
        square_sum = torch.sum(embeddings ** 2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        deep_output = x
        for layer in self.deep:
            deep_output = layer(deep_output)
        deep_output = self.deep_output(deep_output)
        
        output = linear_part + fm_second_order + deep_output + self.bias
        return torch.sigmoid(output)


class DeepFMModelLoader:
    """DeepFM 模型加载器"""
    
    def __init__(self, model_path: str, feature_meta_path: str, model_version: str = None):
        """
        初始化模型加载器
        
        Args:
            model_path: DeepFM 模型文件路径（.pt 文件）
            feature_meta_path: 特征元数据文件路径
            model_version: 模型版本（可选）
        """
        self.model_path = model_path
        self.feature_meta_path = feature_meta_path
        self.model_version = model_version
        self.model = None
        self.feature_columns = None
        self.feature_count = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        """加载模型和特征元数据"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if not os.path.exists(self.feature_meta_path):
            raise FileNotFoundError(f"特征元数据文件不存在: {self.feature_meta_path}")
        
        try:
            # 加载特征元数据
            with open(self.feature_meta_path, "r") as f:
                meta = json.load(f)
            
            self.feature_columns = meta.get("feature_columns", [])
            self.feature_count = meta.get("feature_count", len(self.feature_columns))
            self.model_version = meta.get("model_version", self.model_version)
            
            # 加载模型
            logger.info(f"正在加载模型: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            model_config = checkpoint['model_config']
            self.model = DeepFM(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            # 加载标准化器参数
            self.scaler_mean = np.array(checkpoint.get('scaler_mean', []))
            self.scaler_scale = np.array(checkpoint.get('scaler_scale', []))
            
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"模型版本: {self.model_version or 'unknown'}")
            logger.info(f"特征数量: {self.feature_count}")
            logger.info(f"使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise
    
    def _validate_features(self, features: dict) -> dict:
        """验证和标准化特征"""
        validated = {}
        missing_features = []
        invalid_features = []
        
        for col in self.feature_columns:
            if col not in features:
                missing_features.append(col)
                validated[col] = 0.0
            else:
                value = features[col]
                try:
                    validated[col] = float(value)
                    if not np.isfinite(validated[col]):
                        invalid_features.append(col)
                        validated[col] = 0.0
                except (ValueError, TypeError):
                    invalid_features.append(col)
                    validated[col] = 0.0
        
        if missing_features:
            logger.warning(f"缺失特征（将使用默认值 0.0）: {missing_features}")
        if invalid_features:
            logger.warning(f"无效特征（将使用默认值 0.0）: {invalid_features}")
        
        return validated
    
    def _normalize_features(self, features: dict) -> dict:
        """特征标准化"""
        if self.scaler_mean is None or len(self.scaler_mean) == 0:
            return features
        
        normalized = features.copy()
        for i, col in enumerate(self.feature_columns):
            if col in normalized and i < len(self.scaler_mean):
                mean = self.scaler_mean[i]
                scale = self.scaler_scale[i] if i < len(self.scaler_scale) else 1.0
                if scale > 0:
                    normalized[col] = (normalized[col] - mean) / scale
        
        return normalized
    
    def predict(self, features_list: list[dict]) -> list[float]:
        """
        批量预测
        
        Args:
            features_list: 特征字典列表
        
        Returns:
            预测分数列表 (0-1 之间)
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        
        if not isinstance(features_list, list):
            raise ValueError(f"特征必须是列表类型，当前类型: {type(features_list)}")
        
        if len(features_list) == 0:
            return []
        
        # 批量验证和标准化特征
        feature_vectors = []
        for features in features_list:
            if not isinstance(features, dict):
                raise ValueError(f"特征列表中的元素必须是字典类型，当前类型: {type(features)}")
            
            validated_features = self._validate_features(features)
            normalized_features = self._normalize_features(validated_features)
            
            feature_vector = [normalized_features[col] for col in self.feature_columns]
            feature_vectors.append(feature_vector)
        
        # 转换为 tensor
        X = torch.FloatTensor(feature_vectors).to(self.device)
        
        # 批量预测
        with torch.no_grad():
            scores = self.model(X)
            scores_list = [float(score.item()) for score in scores]
        
        logger.debug(f"批量预测完成，样本数: {len(scores_list)}")
        return scores_list
