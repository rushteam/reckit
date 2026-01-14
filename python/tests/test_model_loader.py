"""
模型加载器单元测试
"""
import json
import os
import tempfile
import unittest

import numpy as np
import xgboost as xgb

# 添加项目根目录到路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from service.model_loader import ModelLoader


class TestModelLoader(unittest.TestCase):
    """模型加载器测试"""
    
    def setUp(self):
        """测试前准备：创建临时模型和元数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.json")
        self.meta_path = os.path.join(self.temp_dir, "test_meta.json")
        
        # 创建简单的测试模型
        self.feature_columns = ["ctr", "cvr", "price"]
        X_train = np.array([[0.1, 0.05, 100], [0.2, 0.08, 200]])
        y_train = np.array([0, 1])
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(
            {"objective": "binary:logistic", "max_depth": 2},
            dtrain,
            num_boost_round=5
        )
        model.save_model(self.model_path)
        
        # 创建特征元数据
        meta = {
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "model_version": "test_v1",
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_model(self):
        """测试模型加载"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        self.assertIsNotNone(loader.model)
        self.assertEqual(loader.feature_columns, self.feature_columns)
        self.assertEqual(loader.feature_count, len(self.feature_columns))
        self.assertEqual(loader.model_version, "test_v1")
    
    def test_load_model_not_found(self):
        """测试模型文件不存在"""
        loader = ModelLoader("nonexist.json", self.meta_path)
        with self.assertRaises(FileNotFoundError):
            loader.load()
    
    def test_predict(self):
        """测试预测"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        features = {"ctr": 0.15, "cvr": 0.08, "price": 99.0}
        score = loader.predict(features)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_predict_missing_features(self):
        """测试缺失特征（应该使用默认值 0.0）"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        # 只提供部分特征
        features = {"ctr": 0.15}
        score = loader.predict(features)
        
        self.assertIsInstance(score, float)
    
    def test_predict_invalid_features(self):
        """测试无效特征（应该使用默认值 0.0）"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        # 提供无效特征值
        features = {"ctr": "invalid", "cvr": None, "price": np.nan}
        score = loader.predict(features)
        
        self.assertIsInstance(score, float)
    
    def test_predict_model_not_loaded(self):
        """测试模型未加载时预测"""
        loader = ModelLoader(self.model_path, self.meta_path)
        
        with self.assertRaises(RuntimeError):
            loader.predict({"ctr": 0.15})
    
    def test_validate_features(self):
        """测试特征验证"""
        loader = ModelLoader(self.model_path, self.meta_path)
        loader.load()
        
        # 完整特征
        features = {"ctr": 0.15, "cvr": 0.08, "price": 99.0}
        validated = loader._validate_features(features)
        self.assertEqual(len(validated), len(self.feature_columns))
        
        # 缺失特征
        features = {"ctr": 0.15}
        validated = loader._validate_features(features)
        self.assertEqual(len(validated), len(self.feature_columns))
        self.assertEqual(validated["cvr"], 0.0)
        self.assertEqual(validated["price"], 0.0)


if __name__ == "__main__":
    unittest.main()
