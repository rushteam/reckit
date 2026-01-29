"""
TorchServe 协议约定（与 reckit 约束一致）

路径：POST /predictions/{model_name}
请求：{"data": [{"feature_a": 0.1, ...}, ...]}
响应：{"predictions": [...]}（标量列表或 embedding 展平或对象列表）
健康：GET /ping -> {"status": "Healthy"}
"""
from pydantic import BaseModel


class TorchServePredictRequest(BaseModel):
    """统一请求体：{"data": [{"feature_a": 0.1, ...}, ...]}"""
    data: list[dict[str, float]] = []


class TorchServePredictResponse(BaseModel):
    """统一响应体：{"predictions": [...]}（标量/embedding/对象列表）"""
    predictions: list  # list[float] 或 list[dict]（如 MMoE 的 ctr/watch_time/gmv）
