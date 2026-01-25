"""
Prometheus 指标：推理服务可观测性

- model_predict_requests_total：/predict 请求总数（按 status 分桶）
- model_predict_duration_seconds：/predict 耗时直方图
- model_reload_total：/reload 调用次数（按 status 分桶）
- model_version_info：当前加载的模型版本（Gauge，label=version）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    Counter = Histogram = Gauge = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from starlette.responses import Response

PREDICT_REQUESTS = (
    Counter(
        "model_predict_requests_total",
        "Total /predict requests",
        ["status"],
    )
    if Counter is not None
    else None
)
PREDICT_LATENCY = (
    Histogram(
        "model_predict_duration_seconds",
        "Predict request duration in seconds",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    if Histogram is not None
    else None
)
RELOAD_TOTAL = (
    Counter(
        "model_reload_total",
        "Total /reload calls",
        ["status"],
    )
    if Counter is not None
    else None
)
MODEL_VERSION_INFO = (
    Gauge(
        "model_version_info",
        "Loaded model version (1 = loaded, labels hold version)",
        ["version"],
    )
    if Gauge is not None
    else None
)


def metrics_response() -> "Response":
    """返回 Prometheus 文本格式的 /metrics 响应。"""
    from fastapi import Response

    if generate_latest is None:
        return Response(
            content="# prometheus_client not installed\n",
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def set_model_version(version: str | None) -> None:
    """更新 model_version_info，当前版本对应 label 置 1。"""
    if MODEL_VERSION_INFO is None:
        return
    v = version or "unknown"
    MODEL_VERSION_INFO.labels(version=v).set(1)


def inc_predict_requests(status: str) -> None:
    if PREDICT_REQUESTS is not None:
        PREDICT_REQUESTS.labels(status=status).inc()


def inc_reload_total(status: str) -> None:
    if RELOAD_TOTAL is not None:
        RELOAD_TOTAL.labels(status=status).inc()


def predict_latency_histogram():
    """返回 PREDICT_LATENCY 的 context manager（with 块内计时）。"""
    if PREDICT_LATENCY is None:
        from contextlib import nullcontext

        return nullcontext()
    return PREDICT_LATENCY.time()
