#!/usr/bin/env python3
"""
XGBoost 模型训练脚本

用法:
    # 本地 CSV（默认）
    python train/train_xgb.py [--version VERSION] [--normalize]
    python train/train_xgb.py --data-source file --data-path data/train_data.csv

    # OSS Parquet（S3/阿里云 OSS/腾讯云 COS/MinIO）
    python train/train_xgb.py --data-source oss --data-path s3://bucket/train.parquet
    python train/train_xgb.py --data-source oss --data-path oss://bucket/train.parquet \\
        --oss-endpoint https://oss-cn-hangzhou.aliyuncs.com

    # MySQL 协议（MySQL / Doris / TiDB）
    python train/train_xgb.py --data-source mysql --doris-query "SELECT * FROM db.table" \\
        --doris-host 127.0.0.1 --doris-port 3306 --doris-user root --doris-password xxx

功能:
    1. 读取训练数据（文件 / OSS Parquet / MySQL 协议）
    2. 切分训练集/验证集
    3. 训练 XGBoost 模型
    4. 保存模型和特征元数据（支持版本管理）
    5. 可选：特征标准化
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入特征配置（使用相对导入）
import importlib.util
features_path = os.path.join(os.path.dirname(__file__), "features.py")
spec = importlib.util.spec_from_file_location("features", features_path)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)
FEATURE_COLUMNS = features.FEATURE_COLUMNS
FEATURE_META_PATH = features.FEATURE_META_PATH
LABEL_COLUMN = features.LABEL_COLUMN
MODEL_DIR = features.MODEL_DIR
MODEL_PATH = features.MODEL_PATH

# 导入数据加载器
_data_loader_path = os.path.join(os.path.dirname(__file__), "data_loader.py")
_data_loader_spec = importlib.util.spec_from_file_location("data_loader", _data_loader_path)
_data_loader = importlib.util.module_from_spec(_data_loader_spec)
_data_loader_spec.loader.exec_module(_data_loader)
load_training_df = _data_loader.load_training_df
get_loader = _data_loader.get_loader


def generate_sample_data(output_path: str, n_samples: int = 1000):
    """生成示例训练数据（列名与 FEATURE_COLUMNS 带前缀格式一致）"""
    np.random.seed(42)
    
    item_ctr = np.random.uniform(0.01, 0.5, n_samples)
    item_cvr = np.random.uniform(0.001, 0.1, n_samples)
    item_price = np.random.uniform(10, 200, n_samples)
    user_age = np.random.randint(18, 60, n_samples).astype(np.float64)
    user_gender = np.random.randint(0, 3, n_samples).astype(np.float64)  # 0=未知，1=男，2=女

    data = {
        "item_ctr": item_ctr,
        "item_cvr": item_cvr,
        "item_price": item_price,
        "user_age": user_age,
        "user_gender": user_gender,
        "cross_age_x_ctr": user_age * item_ctr,
        "cross_gender_x_price": user_gender * item_price,
    }
    
    # 生成标签（简单的线性组合 + 噪声）
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


def train_model(
    data_source: str,
    data_path: str | None = None,
    model_version: str | None = None,
    normalize: bool = False,
    # OSS
    oss_endpoint: str | None = None,
    oss_access_key: str | None = None,
    oss_secret_key: str | None = None,
    oss_region: str | None = None,
    # Doris
    doris_query: str | None = None,
    doris_table: str | None = None,
    doris_database: str = "default",
    doris_host: str | None = None,
    doris_port: int | None = None,
    doris_user: str | None = None,
    doris_password: str | None = None,
):
    """
    训练 XGBoost 模型

    Args:
        data_source: 数据源 file | oss | mysql | doris
        data_path: 数据路径（file: 本地 CSV；oss: s3:// 或 oss:// Parquet）
        model_version: 模型版本（可选）
        normalize: 是否特征标准化
        oss_*: OSS 相关（endpoint、ak、sk、region）
        doris_*: MySQL 协议相关（query/table、database、host、port、user、password，支持 MySQL/Doris/TiDB）
    """
    print(f"数据源: {data_source}, 路径: {data_path or '(query/table)'}")

    # 读取数据
    if data_source.strip().lower() == "file":
        if not data_path:
            raise ValueError("file 数据源需要 --data-path")
        if not os.path.exists(data_path):
            print("数据文件不存在，生成示例数据...")
            df = generate_sample_data(data_path)
        else:
            df = load_training_df("file", path=data_path)
    else:
        df = load_training_df(
            data_source,
            path=data_path,
            query=doris_query,
            table=doris_table,
            database=doris_database,
            host=doris_host,
            port=doris_port,
            user=doris_user,
            password=doris_password,
            endpoint_url=oss_endpoint,
            access_key=oss_access_key,
            secret_key=oss_secret_key,
            region=oss_region,
        )

    print(f"数据形状: {df.shape}")
    print(f"特征列: {FEATURE_COLUMNS}")
    
    # 检查必要的列
    missing_cols = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 准备特征和标签
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values
    
    # 特征标准化（可选）
    scaler = None
    if normalize:
        print("进行特征标准化...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("特征标准化完成")
    
    # 切分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    # 构建 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # XGBoost 参数
    params = {
        "objective": "binary:logistic",  # 二分类
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "n_estimators": 100,
    }
    
    print("\n开始训练 XGBoost 模型...")
    print(f"参数: {params}")
    
    # 训练模型
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10,
    )
    
    # 验证集评估
    y_pred = model.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_val)
    print(f"\n验证集准确率: {accuracy:.4f}")
    print(f"验证集平均预测值: {np.mean(y_pred):.4f}")
    
    # 生成模型版本（如果未指定）
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\n模型已保存: {MODEL_PATH}")
    
    # 保存特征元数据（供推理服务使用）
    feature_meta = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "model_version": model_version,
        "normalized": normalize,
        "created_at": datetime.now().isoformat(),
    }
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"特征元数据已保存: {FEATURE_META_PATH}")
    
    # 保存特征标准化器（如果使用了标准化）
    if scaler is not None:
        scaler_meta = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            scaler_meta[col] = {
                "mean": float(scaler.mean_[i]),
                "std": float(scaler.scale_[i]),
            }
        
        scaler_path = FEATURE_META_PATH.replace("feature_meta.json", "feature_scaler.json")
        with open(scaler_path, "w") as f:
            json.dump(scaler_meta, f, indent=2)
        print(f"特征标准化器已保存: {scaler_path}")
    
    return model, feature_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 XGBoost 模型")
    parser.add_argument("--data-source", default="file", choices=("file", "oss", "mysql", "doris"), help="数据源: file | oss | mysql | doris")
    parser.add_argument("--data-path", default=None, help="数据路径（file: 本地 CSV；oss: s3:// 或 oss:// Parquet）")
    parser.add_argument("--version", type=str, help="模型版本（可选，默认使用时间戳）")
    parser.add_argument("--normalize", action="store_true", help="是否进行特征标准化")
    # OSS
    parser.add_argument("--oss-endpoint", default=None, help="OSS/S3 endpoint（如 https://oss-cn-hangzhou.aliyuncs.com）")
    parser.add_argument("--oss-access-key", default=None, help="OSS/S3 access key（或 env OSS_ACCESS_KEY_ID）")
    parser.add_argument("--oss-secret-key", default=None, help="OSS/S3 secret key（或 env OSS_SECRET_ACCESS_KEY）")
    parser.add_argument("--oss-region", default=None, help="OSS/S3 region（或 env AWS_REGION）")
    # MySQL 协议（MySQL / Doris / TiDB）
    parser.add_argument("--doris-query", default=None, help="MySQL 协议 SQL 查询（与 --doris-table 二选一）")
    parser.add_argument("--doris-table", default=None, help="MySQL 协议表名（与 --doris-query 二选一）")
    parser.add_argument("--doris-database", default="default", help="MySQL 协议库名")
    parser.add_argument("--doris-host", default=None, help="MySQL 协议数据库地址（或 env MYSQL_HOST / DORIS_HOST）")
    parser.add_argument("--doris-port", type=int, default=None, help="MySQL 协议查询端口（MySQL 默认 3306，Doris 默认 9030）")
    parser.add_argument("--doris-user", default=None, help="MySQL 协议用户（或 env MYSQL_USER / DORIS_USER）")
    parser.add_argument("--doris-password", default=None, help="MySQL 协议密码（或 env MYSQL_PASSWORD / DORIS_PASSWORD）")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = args.data_path or os.path.join(data_dir, "train_data.csv")
    if args.data_source == "file" and not args.data_path:
        args.data_path = data_path
    path_for_train = args.data_path if args.data_path else data_path

    try:
        model, feature_meta = train_model(
            data_source=args.data_source,
            data_path=path_for_train,
            model_version=args.version,
            normalize=args.normalize,
            oss_endpoint=args.oss_endpoint,
            oss_access_key=args.oss_access_key,
            oss_secret_key=args.oss_secret_key,
            oss_region=args.oss_region,
            doris_query=args.doris_query,
            doris_table=args.doris_table,
            doris_database=args.doris_database,
            doris_host=args.doris_host,
            doris_port=args.doris_port,
            doris_user=args.doris_user,
            doris_password=args.doris_password,
        )
        print("\n训练完成！")
        print(f"模型版本: {feature_meta['model_version']}")
        print(f"模型文件: {MODEL_PATH}")
        print(f"特征元数据: {FEATURE_META_PATH}")
        print("\n下一步: 启动推理服务")
        print("  cd python")
        print("  uvicorn service.server:app --host 0.0.0.0 --port 8080")
    except Exception as e:
        print(f"训练失败: {e}", file=sys.stderr)
        sys.exit(1)
