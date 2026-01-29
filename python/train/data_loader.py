"""
训练数据加载器

支持多种数据源，统一返回 pd.DataFrame，供 XGBoost、DeepFM 等训练脚本使用。

数据源:
  - file: 本地文件（CSV、Parquet、JSON，与 oss 格式一致）
  - oss:  对象存储（CSV、Parquet、JSON，与 file 格式一致，仅位置不同）
  - mysql: MySQL 协议（支持 MySQL、Doris、TiDB 等）

用法:
  from train.data_loader import get_loader, load_training_df

  loader = get_loader("file", path="data/train_data.csv")
  df = loader.load()

  df = load_training_df("oss", path="s3://bucket/train.parquet", ...)
"""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


def _format_from_path(path: str) -> str:
    """根据路径后缀推断格式：csv / parquet / json，默认 csv。"""
    p = path.lower().split("?")[0]
    if p.endswith(".parquet"):
        return "parquet"
    if p.endswith(".json") or p.endswith(".jsonl"):
        return "json"
    return "csv"


def _resolve_format(path: str, format: str | None) -> str:
    """优先使用 format 参数，为空时按路径后缀识别。"""
    if format:
        f = format.strip().lower()
        if f in ("csv", "parquet", "json"):
            return f
    return _format_from_path(path)


class TrainingDataLoader(ABC):
    """训练数据加载器抽象基类"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """加载训练数据，返回 DataFrame。"""
        pass


class FileDataLoader(TrainingDataLoader):
    """本地文件加载器，支持 CSV、Parquet、JSON（与 oss 支持格式一致）。"""

    def __init__(self, path: str, format: str | None = None, **kwargs: Any) -> None:
        self.path = path
        self.format = format  # 显式指定 csv/parquet/json，为空时按路径后缀识别
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据文件不存在: {self.path}")
        fmt = _resolve_format(self.path, self.format)
        if fmt == "parquet":
            return pd.read_parquet(self.path, **self.kwargs)
        if fmt == "json":
            return pd.read_json(self.path, **self.kwargs)
        return pd.read_csv(self.path, **self.kwargs)


class OSSDataLoader(TrainingDataLoader):
    """
    对象存储数据加载器，支持格式与 file 一致：CSV、Parquet、JSON。

    仅数据位置不同（OSS/S3），路径格式: s3://bucket/key 或 oss://bucket/key。
    支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等 S3 兼容协议。
    """

    def __init__(
        self,
        path: str,
        *,
        format: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.format = format  # 显式指定 csv/parquet/json，为空时按路径后缀识别
        self.endpoint_url = endpoint_url or os.environ.get("OSS_ENDPOINT_URL", "").strip() or None
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("OSS_ACCESS_KEY_ID", ""))
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("OSS_SECRET_ACCESS_KEY", ""))
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            import s3fs
        except ImportError as e:
            raise RuntimeError("读取 OSS/S3 需要安装: pip install s3fs") from e

        path = self.path
        if path.startswith("oss://"):
            path = "s3://" + path[6:]

        client_kwargs: dict[str, Any] = {}
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
            data = f.read()

        fmt = _resolve_format(self.path, self.format)
        if fmt == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise RuntimeError("读取 Parquet 需要安装: pip install pyarrow") from e
            table = pq.read_table(io.BytesIO(data))
            return table.to_pandas()
        if fmt == "json":
            return pd.read_json(io.BytesIO(data), **self.kwargs)
        kw = dict(self.kwargs)
        encoding = kw.pop("encoding", "utf-8")
        return pd.read_csv(io.BytesIO(data), encoding=encoding, **kw)


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
    format: str | None = None,
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

    - file:  path 必填；format 可选，指定 csv/parquet/json，为空时按路径后缀识别。
    - oss:   path 必填；format 可选，同上；可选 endpoint_url, access_key, secret_key, region。
    - mysql: query 或 table 必填；可选 database, host, port, user, password（支持 MySQL、Doris、TiDB 等）。
    - doris: 同 mysql。
    """
    s = (source or "").strip().lower()
    if s == "file":
        if not path:
            raise ValueError("file 数据源需要 path")
        kw = dict(kwargs)
        kw.pop("format", None)
        return FileDataLoader(path, format=format, **kw)
    if s == "oss":
        if not path:
            raise ValueError("oss 数据源需要 path（如 s3://bucket/key.csv 或 key.parquet）")
        kw = dict(kwargs)
        kw.pop("format", None)
        return OSSDataLoader(
            path,
            format=format,
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            **kw,
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
    format: str | None = None,
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
    """根据数据源加载训练数据，返回 DataFrame。format 可选，指定 csv/parquet/json，为空时按路径后缀识别。"""
    loader = get_loader(
        source,
        path=path,
        format=format,
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
