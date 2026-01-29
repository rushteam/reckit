#!/usr/bin/env python3
"""
MMoE 多目标推理服务

提供 /predict 多任务输出 (ctr, watch_time, gmv)，供 Golang MMoERerankNode 调用。

启动:
    uvicorn service.mmoe_server:app --host 0.0.0.0 --port 8081 --timeout-keep-alive 30
"""
import importlib.util
import logging
import os
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

loader_path = os.path.join(os.path.dirname(__file__), "mmoe_model_loader.py")
spec = importlib.util.spec_from_file_location("mmoe_model_loader", loader_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MMoEModelLoader = mod.MMoEModelLoader

from service.middleware import RequestIDMiddleware
from service import metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MMOE_MODEL_PATH = os.path.join(MODEL_DIR, "mmoe_model.pt")
MMOE_FEATURE_META_PATH = os.path.join(MODEL_DIR, "mmoe_feature_meta.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
PORT = int(os.getenv("PORT", "8081"))
HOST = os.getenv("HOST", "0.0.0.0")
RELOAD_API_KEY = os.getenv("RELOAD_API_KEY", "").strip() or None

app = FastAPI(
    title="Reckit MMoE Model Service",
    description="MMoE 多目标模型 (CTR / watch_time / GMV)，供 MMoERerankNode 调用",
    version="1.0.0",
)
app.add_middleware(RequestIDMiddleware)

model_loader = None
model_lock = threading.RLock()


def _check_reload_auth(request: Request) -> None:
    if not RELOAD_API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != RELOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.on_event("startup")
async def startup_event():
    global model_loader
    try:
        logger.info("正在启动 MMoE 模型服务...")
        model_loader = MMoEModelLoader(MMOE_MODEL_PATH, MMOE_FEATURE_META_PATH, MODEL_VERSION)
        model_loader.load()
        metrics.set_model_version(model_loader.model_version)
        logger.info("MMoE 模型服务启动成功")
    except FileNotFoundError as e:
        logger.error("模型未找到: %s，请先运行 python train/train_mmoe.py", e)
        raise
    except Exception as e:
        logger.error("MMoE 加载失败: %s", e, exc_info=True)
        raise


@app.get("/")
async def root():
    return {
        "service": "Reckit MMoE Model Service",
        "status": "running",
        "model_loaded": model_loader is not None,
        "model_version": model_loader.model_version if model_loader else None,
        "tasks": ["ctr", "watch_time", "gmv"],
    }


@app.get("/health")
async def health():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/ping")
async def ping():
    """TorchServe 风格健康检查，与 TorchServe Inference API 一致：返回 {"status": "Healthy"}。"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "Healthy"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics.metrics_response()


@app.post("/reload")
async def reload_model(request: Request):
    try:
        _check_reload_auth(request)
    except HTTPException as e:
        if e.status_code == 401:
            metrics.inc_reload_total("401")
        raise
    global model_loader
    with model_lock:
        try:
            new_loader = MMoEModelLoader(MMOE_MODEL_PATH, MMOE_FEATURE_META_PATH, MODEL_VERSION)
            new_loader.load()
            old_v = model_loader.model_version if model_loader else None
            model_loader = new_loader
            metrics.set_model_version(model_loader.model_version)
            metrics.inc_reload_total("success")
            return {"status": "success", "old_version": old_v, "new_version": model_loader.model_version}
        except FileNotFoundError:
            metrics.inc_reload_total("404")
            raise HTTPException(status_code=404, detail="Model file not found")
        except HTTPException:
            raise
        except Exception as e:
            metrics.inc_reload_total("500")
            logger.error("MMoE 重载失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    """兼容 features_list（旧）与 data（统一协议）。"""
    features_list: list[dict[str, float]] = []
    data: list[dict[str, float]] = []


class TaskScores(BaseModel):
    ctr: float
    watch_time: float
    gmv: float


class PredictResponse(BaseModel):
    """兼容 scores_list（旧）与 predictions（统一协议），二者同源。"""
    scores_list: list[TaskScores] = []
    predictions: list[TaskScores] = []


class TorchServePredictRequest(BaseModel):
    """统一请求体：{"data": [{"feature_a": 0.1, ...}, ...]}"""
    data: list[dict[str, float]] = []


class TorchServePredictResponse(BaseModel):
    """统一响应体：{"predictions": [{"ctr", "watch_time", "gmv"}, ...]}（多任务每条为对象）"""
    predictions: list[TaskScores]


def _predict_multi_task(features_list: list[dict[str, float]]):
    with model_lock:
        return model_loader.predict_multi_task(features_list)


@app.post("/predictions/{model_name}", response_model=TorchServePredictResponse)
async def predictions_unified(model_name: str, request: TorchServePredictRequest):
    """统一协议：POST /predictions/{model_name}，请求 {"data": [...]}，响应 {"predictions": [{"ctr","watch_time","gmv"}, ...]}。模型名建议 mmoe。"""
    if model_loader is None:
        metrics.inc_predict_requests("503")
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.data:
        raise HTTPException(status_code=400, detail="data 不能为空")
    status = "200"
    try:
        with metrics.predict_latency_histogram():
            out = _predict_multi_task(request.data)
            predictions = [TaskScores(ctr=x["ctr"], watch_time=x["watch_time"], gmv=x["gmv"]) for x in out]
        metrics.inc_predict_requests(status)
        return TorchServePredictResponse(predictions=predictions)
    except ValueError as e:
        metrics.inc_predict_requests("400")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics.inc_predict_requests("500")
        logger.exception("predictions failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """兼容旧协议 features_list/scores_list 与统一协议 data/predictions。"""
    if model_loader is None:
        metrics.inc_predict_requests("503")
        raise HTTPException(status_code=503, detail="Model not loaded")
    features_list = request.data if request.data else request.features_list
    if not features_list:
        raise HTTPException(status_code=400, detail="features_list 或 data 不能为空")
    status = "200"
    try:
        with metrics.predict_latency_histogram():
            out = _predict_multi_task(features_list)
            lst = [TaskScores(ctr=x["ctr"], watch_time=x["watch_time"], gmv=x["gmv"]) for x in out]
        metrics.inc_predict_requests(status)
        return PredictResponse(scores_list=lst, predictions=lst)
    except ValueError as e:
        status = "400"
        metrics.inc_predict_requests(status)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        status = "500"
        metrics.inc_predict_requests(status)
        logger.error("MMoE 预测失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 MMoE 服务 http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
