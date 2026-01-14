"""
集成测试：测试完整的训练和推理流程
"""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import xgboost as xgb

# 添加项目根目录到路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from service.model_loader import ModelLoader


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.json")
        self.meta_path = os.path.join(self.temp_dir, "meta.json")
        
        # 生成训练数据
        np.random.seed(42)
        n_samples = 100
        data = {
            "ctr": np.random.uniform(0.01, 0.5, n_samples),
            "cvr": np.random.uniform(0.001, 0.1, n_samples),
            "price": np.random.uniform(10, 200, n_samples),
        }
        label = (
            0.5 * data["ctr"] * 10 +
            0.3 * data["cvr"] * 20 +
            np.random.normal(0, 0.1, n_samples)
        )
        data["label"] = (label > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        
        # 训练模型
        feature_columns = ["ctr", "cvr", "price"]
        X = df[feature_columns].values
        y = df["label"].values
        
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3},
            dtrain,
            num_boost_round=10
        )
        model.save_model(self.model_path)
        
        # 保存元数据
        meta = {
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "model_version": "integration_test_v1",
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """测试完整的训练-加载-预测流程"""
        # 加载模型
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        # 预测
        test_features = [
            {"ctr": 0.15, "cvr": 0.08, "price": 99.0},
            {"ctr": 0.25, "cvr": 0.12, "price": 150.0},
            {"ctr": 0.05, "cvr": 0.02, "price": 50.0},
        ]
        
        scores = []
        for features in test_features:
            score = loader.predict(features)
            scores.append(score)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # 验证预测结果的一致性（高 CTR/CVR 应该有更高的分数）
        self.assertGreater(scores[1], scores[2])  # 高特征 > 低特征
    
    def test_feature_validation(self):
        """测试特征验证功能"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        # 测试各种特征情况
        test_cases = [
            ({"ctr": 0.15, "cvr": 0.08, "price": 99.0}, True),  # 完整特征
            ({"ctr": 0.15}, True),  # 缺失特征
            ({"ctr": "invalid", "cvr": None}, True),  # 无效特征
            ({}, True),  # 空特征
        ]
        
        for features, should_succeed in test_cases:
            if should_succeed:
                score = loader.predict(features)
                self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
