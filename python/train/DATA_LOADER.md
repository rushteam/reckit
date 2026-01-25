# 训练数据加载器

训练脚本（`train_xgb.py`、`train_deepfm.py`）支持多种数据源，除本地文件外，可从 **OSS Parquet** 或 **Doris** 读取训练数据。

## 数据源

| 数据源 | 说明 | 格式 |
|--------|------|------|
| **file** | 本地文件 | CSV |
| **oss** | 对象存储 Parquet | S3/OSS Parquet（`s3://` 或 `oss://`） |
| **mysql** | MySQL 协议 | 支持 MySQL、Doris、TiDB 等，SQL 查询或表名 |
| **doris** | （别名） | 同 mysql，向后兼容 |

## 用法

### 1. 本地文件（默认）

```bash
# 默认 data/train_data.csv
python train/train_xgb.py

# 指定路径
python train/train_xgb.py --data-source file --data-path data/train_data.csv
python train/train_deepfm.py --data-source file --data-path data/train_data.csv
```

### 2. OSS Parquet（S3 / 阿里云 OSS / 腾讯云 COS / MinIO）

```bash
# AWS S3（使用默认 region / 环境变量）
python train/train_xgb.py --data-source oss --data-path s3://my-bucket/train/dt=2025-01-23/data.parquet

# 阿里云 OSS（需 endpoint）
python train/train_xgb.py --data-source oss --data-path oss://my-bucket/train/data.parquet \
  --oss-endpoint https://oss-cn-hangzhou.aliyuncs.com

# 使用环境变量
export OSS_ENDPOINT_URL=https://oss-cn-hangzhou.aliyuncs.com
export OSS_ACCESS_KEY_ID=xxx
export OSS_SECRET_ACCESS_KEY=xxx
python train/train_xgb.py --data-source oss --data-path oss://my-bucket/train.parquet
```

**OSS 参数**：

| 参数 | 环境变量 | 说明 |
|------|----------|------|
| `--oss-endpoint` | `OSS_ENDPOINT_URL` | OSS/S3 endpoint（如 OSS 阿里云） |
| `--oss-access-key` | `AWS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_ID` | Access Key |
| `--oss-secret-key` | `AWS_SECRET_ACCESS_KEY` / `OSS_SECRET_ACCESS_KEY` | Secret Key |
| `--oss-region` | `AWS_REGION` | Region（默认 `us-east-1`） |

**依赖**：`pip install pyarrow s3fs`

### 3. MySQL 协议（MySQL / Doris / TiDB）

```bash
# SQL 查询（MySQL）
python train/train_xgb.py --data-source mysql \
  --doris-query "SELECT * FROM db.train_table WHERE dt='2025-01-23'" \
  --doris-host 127.0.0.1 --doris-port 3306 --doris-user root --doris-password xxx \
  --doris-database db

# SQL 查询（Doris）
python train/train_xgb.py --data-source doris \
  --doris-query "SELECT * FROM db.train_table WHERE dt='2025-01-23'" \
  --doris-host 127.0.0.1 --doris-port 9030 --doris-user root --doris-password xxx \
  --doris-database db

# 表名（SELECT * FROM database.table）
python train/train_xgb.py --data-source mysql --doris-table train_table \
  --doris-database db --doris-host 127.0.0.1 --doris-port 3306 \
  --doris-user root --doris-password xxx
```

**MySQL 协议参数**（`--doris-*` 参数名保留用于向后兼容）：

| 参数 | 环境变量 | 说明 |
|------|----------|------|
| `--doris-query` | - | SQL 查询（与 `--doris-table` 二选一） |
| `--doris-table` | - | 表名（与 `--doris-query` 二选一） |
| `--doris-database` | - | 库名（默认 `default`） |
| `--doris-host` | `MYSQL_HOST` / `DORIS_HOST` | 数据库地址 |
| `--doris-port` | `MYSQL_PORT` / `DORIS_PORT` | 查询端口（MySQL 默认 3306，Doris 默认 9030） |
| `--doris-user` | `MYSQL_USER` / `DORIS_USER` | 用户 |
| `--doris-password` | `MYSQL_PASSWORD` / `DORIS_PASSWORD` | 密码 |

**依赖**：`pip install pymysql`

## 数据格式要求

与本地 CSV 一致：

- **特征列**：与 `train/features.py` 中 `FEATURE_COLUMNS` 一致（如 `item_ctr`, `item_cvr`, `user_age`, ...）
- **标签列**：`label`（0/1 二分类）

Parquet、MySQL/Doris 表需包含相同列名即可。

## 在代码中使用

```python
from train.data_loader import get_loader, load_training_df

# 方式 1：工厂函数
loader = get_loader("file", path="data/train_data.csv")
df = loader.load()

# 方式 2：一步加载
df = load_training_df("oss", path="s3://bucket/train.parquet", endpoint_url="https://...")
df = load_training_df("mysql", query="SELECT * FROM db.t", host="127.0.0.1", port=3306, ...)
df = load_training_df("doris", query="SELECT * FROM db.t", host="127.0.0.1", port=9030, ...)  # 向后兼容
```

## 依赖汇总

| 数据源 | 额外依赖 |
|--------|----------|
| file | 无（pandas 已包含） |
| oss | `pyarrow`, `s3fs` |
| mysql / doris | `pymysql` |

全部安装：`pip install -r requirements.txt`
