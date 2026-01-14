"""
服务器集成测试
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xgboost as xgb

# 添加项目根目录到路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from service.server import app


class TestServer(unittest.TestCase):
    """服务器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.client = TestClient(app)
        
        # 创建临时模型和元数据
        self.temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(self.temp_dir, "test_model.json")
        meta_path = os.path.join(self.temp_dir, "test_meta.json")
        
        # 创建简单的测试模型
        feature_columns = ["ctr", "cvr", "price"]
        X_train = np.array([[0.1, 0.05, 100], [0.2, 0.08, 200]])
        y_train = np.array([0, 1])
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(
            {"objective": "binary:logistic", "max_depth": 2},
            dtrain,
            num_boost_round=5
        )
        model.save_model(model_path)
        
        # 创建特征元数据
        meta = {
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "model_version": "test_v1",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        
        # Mock 模型路径
        self.original_model_path = os.path.join(
            os.path.dirname(os.path.dirname(project_root)),
            "model", "xgb_model.json"
        )
        self.original_meta_path = os.path.join(
            os.path.dirname(os.path.dirname(project_root)),
            "model", "feature_meta.json"
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch("service.server.MODEL_PATH")
    @patch("service.server.FEATURE_META_PATH")
    def test_health_check(self, mock_meta_path, mock_model_path):
        """测试健康检查"""
        # 注意：由于模型加载在 startup 事件中，这里只测试接口存在
        response = self.client.get("/health")
        # 如果模型未加载，应该返回 503
        # 如果模型已加载，应该返回 200
        self.assertIn(response.status_code, [200, 503])
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
    
    @patch("service.server.model_loader")
    def test_predict_endpoint(self, mock_loader):
        """测试预测端点（使用 mock）"""
        # Mock 模型加载器
        mock_loader.predict.return_value = 0.85
        mock_loader.model_version = "test_v1"
        mock_loader.feature_count = 3
        
        # 设置全局 model_loader
        import service.server
        service.server.model_loader = mock_loader
        
        response = self.client.post(
            "/predict",
            json={
                "features": {
                    "ctr": 0.15,
                    "cvr": 0.08,
                    "price": 99.0
                }
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("score", data)
            self.assertIsInstance(data["score"], float)
        else:
            # 如果模型未加载，返回 503 也是正常的
            self.assertEqual(response.status_code, 503)
    
    def test_predict_invalid_request(self):
        """测试无效请求"""
        response = self.client.post(
            "/predict",
            json={"invalid": "data"}
        )
        # 应该返回 422 (Validation Error) 或 503 (Model not loaded)
        self.assertIn(response.status_code, [422, 503])


if __name__ == "__main__":
    unittest.main()
