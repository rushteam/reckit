#!/usr/bin/env python3
"""
统一模型推理服务：单进程多模型，按 model_name 分发。

路由：POST /predictions/{model_name}，请求 {"data": [...]}，响应 {"predictions": [...]}。
双塔通过 model_name == "user_tower" 与 "item_tower" 区分；其余模型名：xgb, deepfm, mmoe, youtube_dnn, dssm, graph_recall。

环境变量：
- ENABLED_MODELS：逗号分隔，如 "xgb,deepfm,mmoe,user_tower,item_tower,youtube_dnn,dssm,graph_recall"
- PORT / HOST：同各单模型服务

启动：
    uvicorn service.unified_server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30
"""
import json
import logging
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 项目根
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from service.domain.protocol import TorchServePredictResponse
from service.middleware import RequestIDMiddleware
from service import metrics

# 模型目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")
ENABLED_MODELS_STR = os.getenv("ENABLED_MODELS", "xgb").strip()
ENABLED_MODELS = [s.strip() for s in ENABLED_MODELS_STR.split(",") if s.strip()]


class UnifiedPredictRequest(BaseModel):
    """统一请求体：data 中每项可为特征 dict（含 float）或 youtube_dnn/graph_recall 专用结构。"""
    data: list[dict[str, Any]] = []


# ---------- 后端抽象：load + predict ----------
class ModelBackend:
    def load(self) -> None:
        raise NotImplementedError

    def predict(self, data: list[dict[str, Any]]) -> list:
        """返回 predictions 列表（float、list[float]、dict、list[str] 等）。"""
        raise NotImplementedError


# ---------- XGB ----------
def _make_xgb_backend() -> ModelBackend:
    from service.model_loader import ModelLoader
    path = os.path.join(MODEL_DIR, "xgb_model.json")
    meta = os.path.join(MODEL_DIR, "feature_meta.json")

    class XGBBackend(ModelBackend):
        def __init__(self):
            self.loader = ModelLoader(path, meta)

        def load(self) -> None:
            self.loader.load()

        def predict(self, data: list[dict[str, Any]]) -> list:
            return self.loader.predict([{k: float(v) for k, v in d.items()} for d in data])
    return XGBBackend()


# ---------- DeepFM ----------
def _make_deepfm_backend() -> ModelBackend:
    from service.deepfm_model_loader import DeepFMModelLoader
    path = os.path.join(MODEL_DIR, "deepfm_model.pt")
    meta = os.path.join(MODEL_DIR, "deepfm_feature_meta.json")

    class DeepFMBackend(ModelBackend):
        def __init__(self):
            self.loader = DeepFMModelLoader(path, meta)

        def load(self) -> None:
            self.loader.load()

        def predict(self, data: list[dict[str, Any]]) -> list:
            return self.loader.predict([{k: float(v) for k, v in d.items()} for d in data])
    return DeepFMBackend()


# ---------- MMoE ----------
def _make_mmoe_backend() -> ModelBackend:
    from service.mmoe_model_loader import MMoEModelLoader
    path = os.path.join(MODEL_DIR, "mmoe_model.pt")
    meta = os.path.join(MODEL_DIR, "mmoe_feature_meta.json")

    class MMoEBackend(ModelBackend):
        def __init__(self):
            self.loader = MMoEModelLoader(path, meta)

        def load(self) -> None:
            self.loader.load()

        def predict(self, data: list[dict[str, Any]]) -> list:
            return self.loader.predict_multi_task([{k: float(v) for k, v in d.items()} for d in data])
    return MMoEBackend()


# ---------- Two Tower：user_tower / item_tower 共用同一 loader ----------
def _make_two_tower_backends() -> tuple[ModelBackend, ModelBackend]:
    from service.two_tower_model_loader import TwoTowerModelLoader
    path = os.path.join(MODEL_DIR, "two_tower_model.pt")
    meta = os.path.join(MODEL_DIR, "two_tower_meta.json")
    loader = TwoTowerModelLoader(path, meta)
    loader.load()

    class UserTowerBackend(ModelBackend):
        def __init__(self, ldr: Any):
            self.loader = ldr

        def load(self) -> None:
            pass  # 已在外层 load

        def predict(self, data: list[dict[str, Any]]) -> list:
            features_list = [{k: float(v) for k, v in d.items()} for d in data]
            return self.loader.predict_user_embeddings(features_list)

    class ItemTowerBackend(ModelBackend):
        def __init__(self, ldr: Any):
            self.loader = ldr

        def load(self) -> None:
            pass

        def predict(self, data: list[dict[str, Any]]) -> list:
            return [self.loader.get_item_embedding({k: float(v) for k, v in d.items()}) for d in data]

    return UserTowerBackend(loader), ItemTowerBackend(loader)


# ---------- YouTube DNN ----------
def _make_youtube_dnn_backend() -> ModelBackend:
    from service.youtube_dnn_model_loader import YouTubeDNNModelLoader
    path = os.path.join(MODEL_DIR, "youtube_dnn_model.pt")
    meta = os.path.join(MODEL_DIR, "youtube_dnn_meta.json")

    class YouTubeDNNBackend(ModelBackend):
        def __init__(self):
            self.loader = YouTubeDNNModelLoader(path, meta)

        def load(self) -> None:
            self.loader.load()

        def predict(self, data: list[dict[str, Any]]) -> list:
            out = []
            for d in data:
                uf = d.get("user_features") or d
                hist = d.get("history_item_ids") or []
                if isinstance(uf, dict):
                    uf = {k: float(v) for k, v in uf.items()}
                emb = self.loader.get_user_embedding(uf, hist)
                out.append(emb)
            return out
    return YouTubeDNNBackend()


# ---------- DSSM ----------
def _make_dssm_backend() -> ModelBackend:
    from service.dssm_model_loader import DSSMModelLoader
    path = os.path.join(MODEL_DIR, "dssm_model.pt")
    meta = os.path.join(MODEL_DIR, "dssm_meta.json")

    class DSSMBackend(ModelBackend):
        def __init__(self):
            self.loader = DSSMModelLoader(path, meta)

        def load(self) -> None:
            self.loader.load()

        def predict(self, data: list[dict[str, Any]]) -> list:
            return [self.loader.get_query_embedding({k: float(v) for k, v in d.items()}) for d in data]
    return DSSMBackend()


# ---------- Graph Recall ----------
def _make_graph_recall_backend() -> ModelBackend:
    import numpy as np
    meta_path = os.path.join(MODEL_DIR, "node2vec_meta.json")

    class GraphRecallBackend(ModelBackend):
        def __init__(self):
            self._embeddings: dict[str, list[float]] = {}
            self._ids: list[str] = []
            self._mat: np.ndarray | None = None

        def load(self) -> None:
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"嵌入文件不存在: {meta_path}，请先运行 python train/train_node2vec.py")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._embeddings = meta.get("embeddings", {})
            self._ids = list(self._embeddings.keys())
            self._mat = np.array([self._embeddings[n] for n in self._ids], dtype=np.float32) if self._ids else None

        def _recall(self, user_id: str, top_k: int) -> list[str]:
            if user_id not in self._embeddings or self._mat is None:
                return []
            k = max(1, min(500, top_k))
            q = np.array(self._embeddings[user_id], dtype=np.float32).reshape(1, -1)
            dot = np.dot(self._mat, q.T).ravel()
            norm_q = np.linalg.norm(q)
            norm_m = np.linalg.norm(self._mat, axis=1) + 1e-9
            sim = (dot / (norm_m * norm_q)).ravel()
            order = np.argsort(-sim)
            out = []
            for i in order:
                if self._ids[i] == user_id:
                    continue
                out.append(self._ids[i])
                if len(out) >= k:
                    break
            return out

        def predict(self, data: list[dict[str, Any]]) -> list:
            return [
                self._recall(
                    str(d.get("user_id", "")),
                    int(d.get("top_k", 20)),
                )
                for d in data
            ]
    return GraphRecallBackend()


# ---------- 注册表 ----------
REGISTRY: dict[str, ModelBackend] = {}


def _register_backends() -> None:
    global REGISTRY
    REGISTRY = {}
    need_two_tower = "user_tower" in ENABLED_MODELS or "item_tower" in ENABLED_MODELS
    if need_two_tower:
        u, i = _make_two_tower_backends()
        if "user_tower" in ENABLED_MODELS:
            REGISTRY["user_tower"] = u
        if "item_tower" in ENABLED_MODELS:
            REGISTRY["item_tower"] = i
    for name in ENABLED_MODELS:
        if name == "xgb":
            REGISTRY[name] = _make_xgb_backend()
        elif name == "deepfm":
            REGISTRY[name] = _make_deepfm_backend()
        elif name == "mmoe":
            REGISTRY[name] = _make_mmoe_backend()
        elif name in ("user_tower", "item_tower"):
            pass  # 已在 need_two_tower 分支注册
        elif name == "youtube_dnn":
            REGISTRY[name] = _make_youtube_dnn_backend()
        elif name == "dssm":
            REGISTRY[name] = _make_dssm_backend()
        elif name == "graph_recall":
            REGISTRY[name] = _make_graph_recall_backend()
        else:
            logger.warning("未知模型名 %s，已忽略", name)
    for name in ENABLED_MODELS:
        if name in REGISTRY:
            REGISTRY[name].load()
            logger.info("已加载模型: %s", name)


# ---------- App ----------
app = FastAPI(
    title="Reckit Unified Model Service",
    description="统一推理服务，按 model_name 分发；双塔使用 user_tower / item_tower",
    version="1.0.0",
)
app.add_middleware(RequestIDMiddleware)


@app.on_event("startup")
async def startup_event():
    try:
        _register_backends()
        metrics.set_model_version("unified")
        logger.info("统一服务启动成功，已启用: %s", list(REGISTRY.keys()))
    except FileNotFoundError as e:
        logger.error("模型文件未找到: %s", e)
        raise
    except Exception as e:
        logger.error("启动失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {
        "service": "Reckit Unified Model Service",
        "status": "running",
        "models": list(REGISTRY.keys()),
    }


@app.get("/ping")
async def ping():
    """TorchServe 风格健康检查。"""
    return {"status": "Healthy"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics.metrics_response()


def _normalize_predictions(result: list, single_flatten: bool = False) -> list:
    """单条 embedding 时展平为 list[float]，便于 Go 单条调用兼容。"""
    if not single_flatten or len(result) != 1:
        return result
    only = result[0]
    if isinstance(only, list) and only and isinstance(only[0], (int, float)):
        return only
    return result


@app.post("/predictions/{model_name}", response_model=TorchServePredictResponse)
async def predictions_unified(model_name: str, request: UnifiedPredictRequest):
    """
    统一协议：POST /predictions/{model_name}，请求 {"data": [...]}，响应 {"predictions": [...]}。
    双塔：model_name 为 user_tower 或 item_tower。
    """
    if model_name not in REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model_name: {model_name} (enabled: {list(REGISTRY.keys())})",
        )
    if not request.data:
        raise HTTPException(status_code=400, detail="data 不能为空")
    backend = REGISTRY[model_name]
    try:
        with metrics.predict_latency_histogram():
            raw = backend.predict(request.data)
        # 双塔 / embedding 单条时展平为 [f1,f2,...]，与原有 two_tower_server 行为一致
        if model_name in ("user_tower", "item_tower", "youtube_dnn", "dssm"):
            raw = _normalize_predictions(raw, single_flatten=True)
        return TorchServePredictResponse(predictions=raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("predictions failed for model=%s", model_name)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("启动统一模型服务 http://%s:%s，已启用: %s", HOST, PORT, ENABLED_MODELS)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
