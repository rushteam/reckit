"""
训练数据加载器

支持多种数据源，统一返回 pd.DataFrame，供 XGBoost、DeepFM 等训练脚本使用。

数据源:
  - file: 本地文件（CSV）
  - oss:  对象存储 Parquet（AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）
  - mysql: MySQL 协议（支持 MySQL、Doris、TiDB 等）

用法:
  from train.data_loader import get_loader, load_training_df

  loader = get_loader("file", path="data/train_data.csv")
  df = loader.load()

  # 或一步到位
  df = load_training_df("oss", path="s3://bucket/train.parquet", ...)
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class TrainingDataLoader(ABC):
    """训练数据加载器抽象基类"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """加载训练数据，返回 DataFrame。"""
        pass


class FileDataLoader(TrainingDataLoader):
    """本地文件加载器（CSV）"""

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs  # 透传给 pd.read_csv，如 sep, encoding

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")
        return pd.read_csv(self.path, **self.kwargs)


class OSSParquetLoader(TrainingDataLoader):
    """
    对象存储 Parquet 加载器

    支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等 S3 兼容协议。
    路径格式: s3://bucket/key.parquet 或 oss://bucket/key.parquet（需配置 endpoint）。
    """

    def __init__(
        self,
        path: str,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.endpoint_url = endpoint_url or os.environ.get("OSS_ENDPOINT_URL", "").strip() or None
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("OSS_ACCESS_KEY_ID", ""))
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("OSS_SECRET_ACCESS_KEY", ""))
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError("读取 Parquet 需要安装: pip install pyarrow") from e

        # 使用 s3fs（fsspec）支持 S3 / OSS 等；未安装时回退到 fsspec
        use_s3fs = False
        try:
            import s3fs
            use_s3fs = True
        except ImportError:
            pass

        if use_s3fs:
            # oss:// -> s3://，s3fs 用 S3 兼容接口 + endpoint 访问 OSS
            path = self.path
            if path.startswith("oss://"):
                path = "s3://" + path[6:]

            client_kwargs = {}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if self.region:
                client_kwargs["region_name"] = self.region

            fs = s3fs.S3FileSystem(
                key=self.access_key or None,
                secret=self.secret_key or None,
                client_kwargs=client_kwargs if client_kwargs else None,
            )
            with fs.open(path, "rb") as f:
                table = pq.read_table(f)
        else:
            # 本地路径或已挂载的卷
            if self.path.startswith("s3://") or self.path.startswith("oss://"):
                raise RuntimeError(
                    "读取 OSS/S3 Parquet 需要安装: pip install s3fs pyarrow"
                )
            table = pq.read_table(self.path)

        return table.to_pandas()


class MysqlDataLoader(TrainingDataLoader):
    """
    MySQL 协议数据加载器

    支持 MySQL、Apache Doris、TiDB 等使用 MySQL 协议的数据库。
    通过 SQL 查询获取训练数据，返回 DataFrame。
    """

    def __init__(
        self,
        query: str | None = None,
        *,
        table: str | None = None,
        database: str = "default",
        host: str = "127.0.0.1",
        port: int = 9030,
        user: str = "root",
        password: str = "",
        **kwargs: Any,
    ) -> None:
        if not query and not table:
            raise ValueError("MySQL 加载器需要指定 query 或 table")
        self.query = query
        self.table = table
        self.database = database
        self.host = host or os.environ.get("MYSQL_HOST", os.environ.get("DORIS_HOST", "127.0.0.1"))
        self.port = int(port or os.environ.get("MYSQL_PORT", os.environ.get("DORIS_PORT", "9030")))
        self.user = user or os.environ.get("MYSQL_USER", os.environ.get("DORIS_USER", "root"))
        self.password = password or os.environ.get("MYSQL_PASSWORD", os.environ.get("DORIS_PASSWORD", ""))
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            import pymysql
        except ImportError as e:
            raise RuntimeError("连接 MySQL 协议数据库需要安装: pip install pymysql") from e

        sql = self.query
        if not sql and self.table:
            sql = f"SELECT * FROM {self.database}.{self.table}"

        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            **self.kwargs,
        )
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()


def get_loader(
    source: str,
    path: str | None = None,
    query: str | None = None,
    table: str | None = None,
    database: str = "default",
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str | None = None,
    **kwargs: Any,
) -> TrainingDataLoader:
    """
    根据数据源类型返回对应的加载器。

    - file:  path 必填，本地 CSV 路径。
    - oss:   path 必填，如 s3://bucket/key.parquet；可选 endpoint_url, access_key, secret_key, region。
    - mysql: query 或 table 必填；可选 database, host, port, user, password（支持 MySQL、Doris、TiDB 等）。
    - doris: 同 mysql。
    """
    s = (source or "").strip().lower()
    if s == "file":
        if not path:
            raise ValueError("file 数据源需要 path")
        return FileDataLoader(path, **kwargs)
    if s == "oss":
        if not path:
            raise ValueError("oss 数据源需要 path（如 s3://bucket/key.parquet）")
        return OSSParquetLoader(
            path,
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            **kwargs,
        )
    if s == "mysql" or s == "doris":  # doris 作为别名
        return MysqlDataLoader(
            query=query,
            table=table,
            database=database,
            host=host or "127.0.0.1",
            port=port or 9030,
            user=user or "root",
            password=password or "",
            **kwargs,
        )
    raise ValueError(f"不支持的数据源: {source}，可选: file | oss | mysql | doris")


def load_training_df(
    source: str,
    path: str | None = None,
    query: str | None = None,
    table: str | None = None,
    database: str = "default",
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """根据数据源加载训练数据，返回 DataFrame。"""
    loader = get_loader(
        source,
        path=path,
        query=query,
        table=table,
        database=database,
        host=host,
        port=port,
        user=user,
        password=password,
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        **kwargs,
    )
    return loader.load()
