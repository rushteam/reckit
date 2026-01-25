# 数据加载器（Data Loader）

## 概述

`python/train/data_loader.py` 提供了统一的数据加载接口，支持从多种数据源加载训练数据并转换为 `pandas.DataFrame`。

## 支持的数据源

| 数据源 | 类型 | 说明 |
|------|------|------|
| **file** | 本地文件 | 支持 CSV, Parquet, JSON 格式 |
| **oss** | 对象存储 | 支持 S3 兼容协议 (AWS S3, 阿里云 OSS, MinIO 等) |
| **mysql** | 数据库 | 支持 MySQL 协议数据库 |
| **doris** | 数据库 | 别名，同 mysql |

## 核心接口

```python
def load_training_df(source_type, **kwargs) -> pd.DataFrame:
    """
    统一数据加载接口
    :param source_type: 数据源类型 (file, oss, mysql, doris)
    :param kwargs: 各数据源所需的参数
    :return: pandas.DataFrame
    """
```

## 使用示例

### 1. 从本地文件加载

```python
from train.data_loader import load_training_df

# 加载 CSV
df = load_training_df("file", path="data/train.csv")

# 加载 Parquet
df = load_training_df("file", path="data/train.parquet")
```

### 2. 从 OSS 加载

需要配置环境变量或传入参数：`OSS_ENDPOINT`, `OSS_ACCESS_KEY`, `OSS_SECRET_KEY`。

```python
df = load_training_df(
    "oss", 
    bucket="my-bucket", 
    key="data/v1/train.parquet",
    endpoint="http://oss-cn-hangzhou.aliyuncs.com"
)
```

### 3. 从 MySQL/Doris 加载

支持通过 SQL 查询加载数据。

```python
df = load_training_df(
    "mysql", 
    query="SELECT * FROM recommendation.user_behavior WHERE dt = '2026-01-25'",
    host="127.0.0.1",
    port=3306,
    user="root",
    password="password",
    database="recommendation"
)

# 使用 doris 别名
df = load_training_df("doris", query="SELECT * FROM db.t", host="127.0.0.1", port=9030)
```

## MySQL 协议参数

| 参数 | 说明 | 默认值 |
|------|------|------|
| **query** | SQL 查询语句 | 必填 |
| **host** | 数据库地址 | 127.0.0.1 |
| **port** | 端口 | 3306 (mysql) / 9030 (doris) |
| **user** | 用户名 | root |
| **password** | 密码 | - |
| **database** | 数据库名 | - |

## 扩展指南

### 添加新数据源

1. 在 `data_loader.py` 中添加对应的加载函数（如 `_load_from_mongodb`）。
2. 在 `load_training_df` 中注册新类型。
3. 更新相关文档。
