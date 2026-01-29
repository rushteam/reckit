#!/usr/bin/env python3
"""
图召回服务（Node2Vec 嵌入）

加载 node2vec 生成的节点嵌入，提供 /recall：输入 user_id、top_k，返回相似用户 ID 列表。
供 Golang GraphRecall 调用（社交/关注页召回）。

启动:
    uvicorn service.graph_recall_server:app --host 0.0.0.0 --port 8084 --timeout-keep-alive 30
"""
import json
import logging
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from service.middleware import RequestIDMiddleware
from service import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
NODE2VEC_META_PATH = os.path.join(MODEL_DIR, "node2vec_meta.json")
PORT = int(os.getenv("PORT", "8084"))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(title="Reckit Graph Recall", description="Node2Vec 图召回，相似用户", version="1.0.0")
app.add_middleware(RequestIDMiddleware)

_embeddings: dict[str, list[float]] = {}
_ids: list[str] = []
_mat: np.ndarray | None = None


def _load_embeddings() -> None:
    global _embeddings, _ids, _mat
    if not os.path.exists(NODE2VEC_META_PATH):
        raise FileNotFoundError(f"嵌入文件不存在: {NODE2VEC_META_PATH}，请先运行 python train/train_node2vec.py")
    with open(NODE2VEC_META_PATH) as f:
        meta = json.load(f)
    _embeddings = meta.get("embeddings", {})
    _ids = list(_embeddings.keys())
    if not _ids:
        _mat = None
        return
    _mat = np.array([_embeddings[n] for n in _ids], dtype=np.float32)


def _recall(user_id: str, top_k: int) -> list[str]:
    if user_id not in _embeddings or _mat is None:
        return []
    q = np.array(_embeddings[user_id], dtype=np.float32).reshape(1, -1)
    dot = np.dot(_mat, q.T).ravel()
    norm_q = np.linalg.norm(q)
    norm_m = np.linalg.norm(_mat, axis=1) + 1e-9
    sim = (dot / (norm_m * norm_q)).ravel()
    order = np.argsort(-sim)
    out = []
    for i in order:
        if _ids[i] == user_id:
            continue
        out.append(_ids[i])
        if len(out) >= top_k:
            break
    return out


@app.on_event("startup")
async def startup_event():
    try:
        _load_embeddings()
        metrics.set_model_version("node2vec")
        logger.info("图召回服务启动成功, 节点数=%d", len(_embeddings))
    except Exception as e:
        logger.error("加载 Node2Vec 嵌入失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {"service": "Reckit Graph Recall", "status": "running", "nodes": len(_embeddings)}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ping")
async def ping():
    """TorchServe 风格健康检查，与 TorchServe Inference API 一致：返回 {"status": "Healthy"}。"""
    return {"status": "Healthy"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics.metrics_response()


class RecallRequest(BaseModel):
    user_id: str
    top_k: int = 20


class RecallResponse(BaseModel):
    item_ids: list[str]
    scores: list[float] | None = None


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest):
    k = max(1, min(500, req.top_k))
    ids = _recall(req.user_id, k)
    return RecallResponse(item_ids=ids)


if __name__ == "__main__":
    import uvicorn
    logger.info("启动图召回服务 http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
