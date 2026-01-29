#!/usr/bin/env python3
"""
双塔 User Tower 推理服务

协议与 TorchServe 对齐，Go 端统一使用 NewTorchServeClient 调用：
  - GET /ping：健康检查
  - POST /predictions/{model_name}：请求体 {"data": [{"user_age": 0.5, "user_gender": 1, ...}]}，
    响应 {"predictions": [0.1, ..., 0.64]}（User Embedding 向量）

启动:
    uvicorn service.two_tower_server:app --host 0.0.0.0 --port 8085 --timeout-keep-alive 30
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

loader_path = os.path.join(os.path.dirname(__file__), "two_tower_model_loader.py")
spec = importlib.util.spec_from_file_location("two_tower_model_loader", loader_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TwoTowerModelLoader = mod.TwoTowerModelLoader

from service.middleware import RequestIDMiddleware
from service import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TWO_TOWER_MODEL_PATH = os.path.join(MODEL_DIR, "two_tower_model.pt")
TWO_TOWER_META_PATH = os.path.join(MODEL_DIR, "two_tower_meta.json")
PORT = int(os.getenv("PORT", "8085"))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(
    title="Reckit Two-Tower Service",
    description="User Tower 向量服务，供 TwoTowerRecall 调用",
    version="1.0.0",
)
app.add_middleware(RequestIDMiddleware)

model_loader = None


@app.on_event("startup")
async def startup_event():
    global model_loader
    try:
        model_loader = TwoTowerModelLoader(TWO_TOWER_MODEL_PATH, TWO_TOWER_META_PATH)
        model_loader.load()
        metrics.set_model_version("two_tower")
        logger.info("双塔服务启动成功")
    except FileNotFoundError as e:
        logger.error("模型未找到: %s，请先运行 python train/train_two_tower.py", e)
        raise
    except Exception as e:
        logger.error("双塔加载失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {
        "service": "Reckit Two-Tower",
        "status": "running",
        "model_loaded": model_loader is not None,
    }


@app.get("/health")
async def health():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/ping")
async def ping():
    """TorchServe 风格健康检查，与 TorchServe Inference API 一致：返回 {"status": "Healthy"}。"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "Healthy"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics.metrics_response()


class PredictResponse(BaseModel):
    """TorchServe 风格：predictions 为 User Embedding 向量（或标量列表）。"""
    predictions: list[float]


class TorchServePredictRequest(BaseModel):
    """TorchServe 请求体：Go TorchServeClient 发送 {"data": [feature_dict, ...]}。"""
    data: list[dict[str, float]] = []


@app.post("/predictions/{model_name}", response_model=PredictResponse)
async def predictions_torchserve(model_name: str, req: TorchServePredictRequest):
    """
    TorchServe 协议：请求体 {"data": [{"user_age": 0.5, "user_gender": 1, ...}]}，
    响应 {"predictions": [0.1, ..., 0.64]}（单条时为该用户 User Embedding）。
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.data:
        raise HTTPException(status_code=400, detail="data 不能为空")
    try:
        embeddings = model_loader.predict_user_embeddings(req.data)
        if not embeddings:
            raise HTTPException(status_code=400, detail="未返回 embedding")
        emb = embeddings[0]
        return PredictResponse(predictions=emb)
    except Exception as e:
        logger.exception("predictions failed")
        raise HTTPException(status_code=500, detail=str(e))


class UserEmbeddingRequest(BaseModel):
    user_features: dict[str, float]


class UserEmbeddingResponse(BaseModel):
    user_embedding: list[float]


@app.post("/user_embedding", response_model=UserEmbeddingResponse)
async def user_embedding(req: UserEmbeddingRequest):
    """单条用户特征 → User Embedding（便于调试或离线调用）。"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        emb = model_loader.get_user_embedding(req.user_features or {})
        return UserEmbeddingResponse(user_embedding=emb)
    except Exception as e:
        logger.exception("user_embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


class ItemEmbeddingRequest(BaseModel):
    item_features: dict[str, float]


class ItemEmbeddingResponse(BaseModel):
    item_embedding: list[float]


@app.post("/item_embedding", response_model=ItemEmbeddingResponse)
async def item_embedding(req: ItemEmbeddingRequest):
    """单条物品特征 → Item Embedding（用于离线写入向量库）。"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        emb = model_loader.get_item_embedding(req.item_features or {})
        return ItemEmbeddingResponse(item_embedding=emb)
    except Exception as e:
        logger.exception("item_embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("启动双塔服务 http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
