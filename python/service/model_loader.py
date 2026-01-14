"""
模型加载器：封装模型和特征元数据的加载逻辑
"""
import json
import logging
import os
import warnings

import numpy as np
import xgboost as xgb

# 配置日志
logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_path: str, feature_meta_path: str, model_version: str = None):
        """
        初始化模型加载器
        
        Args:
            model_path: XGBoost 模型文件路径
            feature_meta_path: 特征元数据文件路径
            model_version: 模型版本（可选）
        """
        self.model_path = model_path
        self.feature_meta_path = feature_meta_path
        self.model_version = model_version
        self.model = None
        self.feature_columns = None
        self.feature_count = None
        self.feature_scaler = None  # 特征标准化器（可选）
        
    def load(self):
        """加载模型和特征元数据"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if not os.path.exists(self.feature_meta_path):
            raise FileNotFoundError(f"特征元数据文件不存在: {self.feature_meta_path}")
        
        try:
            # 加载模型
            logger.info(f"正在加载模型: {self.model_path}")
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            
            # 加载特征元数据
            with open(self.feature_meta_path, "r") as f:
                meta = json.load(f)
            
            self.feature_columns = meta.get("feature_columns", [])
            self.feature_count = meta.get("feature_count", len(self.feature_columns))
            self.model_version = meta.get("model_version", self.model_version)
            
            # 加载特征标准化器（如果存在）
            scaler_path = self.feature_meta_path.replace("feature_meta.json", "feature_scaler.json")
            if os.path.exists(scaler_path):
                logger.info(f"加载特征标准化器: {scaler_path}")
                with open(scaler_path, "r") as f:
                    scaler_meta = json.load(f)
                    self.feature_scaler = scaler_meta
            
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"模型版本: {self.model_version or 'unknown'}")
            logger.info(f"特征数量: {self.feature_count}")
            logger.debug(f"特征列: {self.feature_columns}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise
        
    def _validate_features(self, features: dict) -> dict:
        """
        验证和标准化特征
        
        Args:
            features: 输入特征字典
        
        Returns:
            验证后的特征字典
        """
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
                    # 检查是否为 NaN 或 Inf
                    if not np.isfinite(validated[col]):
                        invalid_features.append(col)
                        validated[col] = 0.0
                except (ValueError, TypeError):
                    invalid_features.append(col)
                    validated[col] = 0.0
        
        # 记录警告
        if missing_features:
            logger.warning(f"缺失特征（将使用默认值 0.0）: {missing_features}")
        if invalid_features:
            logger.warning(f"无效特征（将使用默认值 0.0）: {invalid_features}")
        
        return validated
    
    def _normalize_features(self, features: dict) -> dict:
        """
        特征标准化（如果配置了标准化器）
        
        Args:
            features: 特征字典
        
        Returns:
            标准化后的特征字典
        """
        if self.feature_scaler is None:
            return features
        
        normalized = features.copy()
        scaler = self.feature_scaler
        
        for col in self.feature_columns:
            if col in normalized and col in scaler:
                # 标准化: (x - mean) / std
                mean = scaler[col].get("mean", 0.0)
                std = scaler[col].get("std", 1.0)
                if std > 0:
                    normalized[col] = (normalized[col] - mean) / std
        
        return normalized
    
    def predict(self, features_list: list[dict]) -> list[float]:
        """
        批量预测
        
        Args:
            features_list: 特征字典列表，例如 [{"ctr": 0.15, "cvr": 0.08, ...}, ...]
        
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
            
            # 验证特征
            validated_features = self._validate_features(features)
            
            # 特征标准化（如果配置了）
            normalized_features = self._normalize_features(validated_features)
            
            # 按特征列顺序构建特征向量
            feature_vector = [normalized_features[col] for col in self.feature_columns]
            feature_vectors.append(feature_vector)
        
        # 转换为 numpy 数组
        X = np.array(feature_vectors)
        
        # 构建 DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # 批量预测
        try:
            scores = self.model.predict(dmatrix)
            scores_list = [float(score) for score in scores]
            logger.debug(f"批量预测完成，样本数: {len(scores_list)}")
            return scores_list
        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            raise
