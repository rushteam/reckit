#!/usr/bin/env python3
"""
模型评估与门控脚本

评估新模型并判断是否通过门控，只有通过门控的模型才能发布。
"""
import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入模型加载器
import importlib.util
loader_path = os.path.join(project_root, "service", "model_loader.py")
spec = importlib.util.spec_from_file_location("model_loader", loader_path)
model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)
ModelLoader = model_loader_module.ModelLoader


def load_model_metrics(version: str, model_dir: str) -> dict:
    """
    加载模型评估指标
    
    Args:
        version: 模型版本
        model_dir: 模型目录
    
    Returns:
        评估指标字典
    """
    # 尝试从训练日志或单独评估文件读取
    metrics_path = os.path.join(model_dir, f"{version}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    # 若不存在，在验证集上重新评估
    print(f"未找到 {version} 的评估指标，在验证集上重新评估...")
    
    # 加载模型
    model_path = os.path.join(model_dir, "xgb_model.json")
    feature_meta_path = os.path.join(model_dir, "feature_meta.json")
    
    if not os.path.exists(model_path) or not os.path.exists(feature_meta_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path} 或 {feature_meta_path}")
    
    loader = ModelLoader(model_path, feature_meta_path, version)
    loader.load()
    
    # 加载验证集（这里简化处理，实际应从数据流水线获取）
    data_path = os.path.join(project_root, "data", "train_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"验证数据不存在: {data_path}")
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(data_path)
    X = df[loader.feature_columns].values
    y = df["label"].values
    
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 预测
    dval = xgb.DMatrix(X_val)
    y_pred = loader.model.predict(dval)
    
    # 计算指标
    auc = roc_auc_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)
    
    metrics = {
        "auc": float(auc),
        "logloss": float(logloss),
        "version": version,
    }
    
    # 保存指标
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_and_gate(new_version: str, current_version: str, model_dir: str) -> bool:
    """
    评估新模型并判断是否通过门控
    
    Args:
        new_version: 新模型版本
        current_version: 当前线上版本
        model_dir: 模型目录
    
    Returns:
        True: 通过门控，可以发布
        False: 未通过门控，不发布
    """
    print(f"评估模型: {new_version} vs {current_version}")
    
    # 加载指标
    try:
        new_metrics = load_model_metrics(new_version, model_dir)
        current_metrics = load_model_metrics(current_version, model_dir)
    except Exception as e:
        print(f"❌ 加载模型指标失败: {e}")
        return False
    
    print(f"\n新模型指标:")
    print(f"  AUC: {new_metrics['auc']:.4f}")
    print(f"  LogLoss: {new_metrics['logloss']:.4f}")
    
    print(f"\n当前模型指标:")
    print(f"  AUC: {current_metrics['auc']:.4f}")
    print(f"  LogLoss: {current_metrics['logloss']:.4f}")
    
    # 门控规则
    auc_threshold = current_metrics['auc'] * 0.99  # 至少不劣化 1%
    logloss_threshold = current_metrics['logloss'] * 1.01  # LogLoss 不增加 1%
    
    if new_metrics['auc'] < auc_threshold:
        print(f"\n❌ 门控失败: AUC 劣化 ({new_metrics['auc']:.4f} < {auc_threshold:.4f})")
        return False
    
    if new_metrics['logloss'] > logloss_threshold:
        print(f"\n❌ 门控失败: LogLoss 增加 ({new_metrics['logloss']:.4f} > {logloss_threshold:.4f})")
        return False
    
    print(f"\n✅ 门控通过: AUC={new_metrics['auc']:.4f}, LogLoss={new_metrics['logloss']:.4f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估与门控")
    parser.add_argument("--version", required=True, help="新模型版本")
    parser.add_argument("--current-version", required=True, help="当前线上版本")
    parser.add_argument("--model-dir", default="/app/model", help="模型目录")
    
    args = parser.parse_args()
    
    if not evaluate_and_gate(args.version, args.current_version, args.model_dir):
        sys.exit(1)
