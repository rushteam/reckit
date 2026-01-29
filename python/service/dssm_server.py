#!/usr/bin/env python3
"""
DSSM Query 向量服务

提供 /query_embedding：输入 query_features，输出 query_embedding。
供 Golang DSSMRecall 调用，再配合 VectorService 做 Doc 检索。

启动:
    uvicorn service.dssm_server:app --host 0.0.0.0 --port 8083 --timeout-keep-alive 30
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

loader_path = os.path.join(os.path.dirname(__file__), "dssm_model_loader.py")
spec = importlib.util.spec_from_file_location("dssm_model_loader", loader_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DSSMModelLoader = mod.DSSMModelLoader

from service.middleware import RequestIDMiddleware
from service import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DSSM_MODEL_PATH = os.path.join(MODEL_DIR, "dssm_model.pt")
DSSM_META_PATH = os.path.join(MODEL_DIR, "dssm_meta.json")
PORT = int(os.getenv("PORT", "8083"))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(title="Reckit DSSM Service", description="Query 向量服务，供 DSSMRecall 调用", version="1.0.0")
app.add_middleware(RequestIDMiddleware)

model_loader = None


@app.on_event("startup")
async def startup_event():
    global model_loader
    try:
        model_loader = DSSMModelLoader(DSSM_MODEL_PATH, DSSM_META_PATH)
        model_loader.load()
        metrics.set_model_version("dssm")
        logger.info("DSSM 服务启动成功")
    except FileNotFoundError as e:
        logger.error("模型未找到: %s，请先运行 python train/train_dssm.py", e)
        raise
    except Exception as e:
        logger.error("DSSM 加载失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {"service": "Reckit DSSM", "status": "running", "model_loaded": model_loader is not None}


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


class QueryEmbeddingRequest(BaseModel):
    query_features: dict[str, float]


class QueryEmbeddingResponse(BaseModel):
    query_embedding: list[float]


@app.post("/query_embedding", response_model=QueryEmbeddingResponse)
async def query_embedding(req: QueryEmbeddingRequest):
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        emb = model_loader.get_query_embedding(req.query_features or {})
        return QueryEmbeddingResponse(query_embedding=emb)
    except Exception as e:
        logger.exception("query_embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 DSSM 服务 http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
