#!/usr/bin/env python3
"""
YouTube DNN 用户向量服务

提供 /user_embedding：输入 user_features + history_item_ids，输出 user_embedding。
供 Golang YouTubeDNNRecall 调用，再配合 VectorService 做 ANN 召回。

启动:
    uvicorn service.youtube_dnn_server:app --host 0.0.0.0 --port 8082 --timeout-keep-alive 30
"""
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

loader_path = os.path.join(os.path.dirname(__file__), "youtube_dnn_model_loader.py")
spec = importlib.util.spec_from_file_location("youtube_dnn_model_loader", loader_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
YouTubeDNNModelLoader = mod.YouTubeDNNModelLoader

from service.middleware import RequestIDMiddleware
from service import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
YOUTUBE_DNN_MODEL_PATH = os.path.join(MODEL_DIR, "youtube_dnn_model.pt")
YOUTUBE_DNN_META_PATH = os.path.join(MODEL_DIR, "youtube_dnn_meta.json")
PORT = int(os.getenv("PORT", "8082"))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(title="Reckit YouTube DNN Service", description="用户向量服务，供 YouTubeDNNRecall 调用", version="1.0.0")
app.add_middleware(RequestIDMiddleware)

model_loader = None


@app.on_event("startup")
async def startup_event():
    global model_loader
    try:
        model_loader = YouTubeDNNModelLoader(YOUTUBE_DNN_MODEL_PATH, YOUTUBE_DNN_META_PATH)
        model_loader.load()
        metrics.set_model_version("youtube_dnn")
        logger.info("YouTube DNN 服务启动成功")
    except FileNotFoundError as e:
        logger.error("模型未找到: %s，请先运行 python train/train_youtube_dnn.py", e)
        raise
    except Exception as e:
        logger.error("YouTube DNN 加载失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {"service": "Reckit YouTube DNN", "status": "running", "model_loaded": model_loader is not None}


@app.get("/health")
async def health():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics.metrics_response()


class UserEmbeddingRequest(BaseModel):
    user_features: dict[str, float]
    history_item_ids: list[str]


class UserEmbeddingResponse(BaseModel):
    user_embedding: list[float]


@app.post("/user_embedding", response_model=UserEmbeddingResponse)
async def user_embedding(req: UserEmbeddingRequest):
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        emb = model_loader.get_user_embedding(req.user_features, req.history_item_ids or [])
        return UserEmbeddingResponse(user_embedding=emb)
    except Exception as e:
        logger.exception("user_embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 YouTube DNN 服务 http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
