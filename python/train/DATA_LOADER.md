# 数据加载器（Data Loader）

## 概述

`python/train/data_loader.py` 提供了统一的数据加载接口，支持从多种数据源加载训练数据并转换为 `pandas.DataFrame`。

## 支持的数据源

| 数据源 | 类型 | 说明 |
|------|------|------|
| **file** | 本地文件 | 支持 CSV、Parquet、JSON；可通过 `format` 指定或按路径后缀推断 |
| **oss** | 对象存储 | 与 file **支持格式一致**，仅位置不同；S3 兼容协议 (AWS S3, 阿里云 OSS, MinIO 等) |
| **mysql** | 数据库 | 支持 MySQL 协议数据库 |
| **doris** | 数据库 | 别名，同 mysql |

## 数据格式参数（file / oss）

- **`format`**（可选）：显式指定 `csv`、`parquet` 或 `json`。
- **为空时**：按 `path` 后缀自动识别（`.csv`、`.parquet`、`.json` / `.jsonl`，其余按 csv 处理）。

适用于无后缀或后缀与真实格式不一致的场景（如 OSS key 无扩展名）。

## 核心接口

```python
def load_training_df(source_type, path=None, format=None, **kwargs) -> pd.DataFrame:
    """
    统一数据加载接口
    :param source_type: 数据源类型 (file, oss, mysql, doris)
    :param path: 文件路径（file/oss 必填）
    :param format: 可选，指定 csv / parquet / json；为空时按 path 后缀识别
    :param kwargs: 各数据源所需的其他参数
    :return: pandas.DataFrame
    """
```

## 使用示例

### 1. 从本地文件加载

```python
from train.data_loader import load_training_df

# 按后缀自动识别
df = load_training_df("file", path="data/train.csv")
df = load_training_df("file", path="data/train.parquet")
df = load_training_df("file", path="data/train.json")

# 显式指定 format（无后缀或后缀与格式不一致时）
df = load_training_df("file", path="data/train", format="csv")
df = load_training_df("file", path="data/train", format="parquet")
```

### 2. 从 OSS 加载

**支持格式与 file 一致**：CSV、Parquet、JSON。可通过 `format` 指定，或为空时按路径后缀推断，仅数据位置在对象存储。

路径使用 `path` 参数，格式为 `oss://bucket/key` 或 `s3://bucket/key`。需要配置环境变量或传入参数：`OSS_ENDPOINT_URL` / `endpoint_url`、`OSS_ACCESS_KEY_ID` / `access_key`、`OSS_SECRET_ACCESS_KEY` / `secret_key`。

```python
# 按后缀识别
df = load_training_df(
    "oss",
    path="oss://my-bucket/data/v1/train.csv",
    endpoint_url="https://oss-cn-hangzhou.aliyuncs.com",
    access_key="your-access-key",
    secret_key="your-secret-key",
)

# 显式指定 format（如 key 无后缀）
df = load_training_df("oss", path="s3://my-bucket/data/v1/train", format="parquet", endpoint_url="...")
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
