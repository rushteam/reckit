#!/usr/bin/env python3
"""
DeepFM 模型推理服务

使用 FastAPI 实现 HTTP 服务，与 Go 端的 RPCModel 协议对齐。
支持 Request ID 中间件、/metrics、/reload 可选鉴权、模型热加载。

启动方式:
    uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30

或者:
    python -m service.deepfm_server
"""
import importlib.util
import logging
import os
import sys
import threading

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入模型加载器
loader_path = os.path.join(os.path.dirname(__file__), "deepfm_model_loader.py")
spec = importlib.util.spec_from_file_location("deepfm_model_loader", loader_path)
model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)
DeepFMModelLoader = model_loader_module.DeepFMModelLoader

from service.middleware import RequestIDMiddleware
from service import metrics

# 模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEEPFM_MODEL_PATH = os.path.join(MODEL_DIR, "deepfm_model.pt")
DEEPFM_FEATURE_META_PATH = os.path.join(MODEL_DIR, "deepfm_feature_meta.json")

# 从环境变量读取配置
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")
RELOAD_API_KEY = os.getenv("RELOAD_API_KEY", "").strip() or None

# 创建 FastAPI 应用
app = FastAPI(
    title="Reckit DeepFM Model Service",
    description="DeepFM 模型推理服务，与 Go 端 RPCModel 协议对齐",
    version="1.0.0",
)
app.add_middleware(RequestIDMiddleware)

# 全局模型加载器
model_loader = None
model_lock = threading.RLock()


def _check_reload_auth(request: Request) -> None:
    """可选 /reload 鉴权：若配置 RELOAD_API_KEY，则校验 X-API-Key 头。"""
    if not RELOAD_API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != RELOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_loader
    try:
        logger.info("正在启动 DeepFM 模型服务...")
        logger.info("模型路径: %s", DEEPFM_MODEL_PATH)
        logger.info("特征元数据路径: %s", DEEPFM_FEATURE_META_PATH)

        model_loader = DeepFMModelLoader(DEEPFM_MODEL_PATH, DEEPFM_FEATURE_META_PATH, MODEL_VERSION)
        model_loader.load()

        metrics.set_model_version(model_loader.model_version)
        logger.info("DeepFM 模型服务启动成功！")
        logger.info("模型版本: %s", model_loader.model_version or "unknown")
        logger.info("特征数量: %d", model_loader.feature_count)
    except FileNotFoundError as e:
        logger.error("模型文件未找到: %s", e)
        logger.error("请先运行训练脚本: python train/train_deepfm.py")
        raise
    except Exception as e:
        logger.error("模型加载失败: %s", e, exc_info=True)
        raise


class PredictRequest(BaseModel):
    """批量预测请求（与 Go RPCModel 协议对齐）"""
    features_list: list[dict[str, float]]


class PredictResponse(BaseModel):
    """批量预测响应（与 Go RPCModel 协议对齐）"""
    scores: list[float]


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Reckit DeepFM Model Service",
        "status": "running",
        "model_loaded": model_loader is not None,
        "model_version": model_loader.model_version if model_loader else None,
        "feature_count": model_loader.feature_count if model_loader else None,
    }


@app.get("/health")
async def health():
    """健康检查"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标端点"""
    return metrics.metrics_response()


@app.post("/reload")
async def reload_model(request: Request):
    """重新加载模型（支持热更新）。若配置 RELOAD_API_KEY，则需 X-API-Key 头。"""
    try:
        _check_reload_auth(request)
    except HTTPException as e:
        if e.status_code == 401:
            metrics.inc_reload_total("401")
        raise

    global model_loader

    with model_lock:
        try:
            logger.info("开始重新加载 DeepFM 模型...")
            logger.info("模型路径: %s", DEEPFM_MODEL_PATH)
            logger.info("特征元数据路径: %s", DEEPFM_FEATURE_META_PATH)

            new_loader = DeepFMModelLoader(DEEPFM_MODEL_PATH, DEEPFM_FEATURE_META_PATH, MODEL_VERSION)
            new_loader.load()

            old_version = model_loader.model_version if model_loader else None
            model_loader = new_loader
            metrics.set_model_version(model_loader.model_version)
            metrics.inc_reload_total("success")

            logger.info("DeepFM 模型重新加载成功！旧版本: %s 新版本: %s", old_version or "unknown", model_loader.model_version or "unknown")

            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "old_version": old_version,
                "new_version": model_loader.model_version,
                "feature_count": model_loader.feature_count,
            }
        except FileNotFoundError as e:
            metrics.inc_reload_total("404")
            logger.error("模型文件未找到: %s", e)
            raise HTTPException(status_code=404, detail=f"Model file not found: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            metrics.inc_reload_total("500")
            logger.error("模型重新加载失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """批量预测接口（与 Go RPCModel 协议对齐）。支持 /metrics 统计。"""
    if model_loader is None:
        metrics.inc_predict_requests("503")
        logger.error("模型未加载，无法进行预测")
        raise HTTPException(status_code=503, detail="Model not loaded")

    status = "200"
    try:
        with metrics.predict_latency_histogram():
            with model_lock:
                logger.debug("收到批量预测请求，样本数: %d", len(request.features_list))
                scores = model_loader.predict(request.features_list)
                logger.debug("批量预测完成，返回分数数量: %d", len(scores))
            metrics.inc_predict_requests(status)
            return PredictResponse(scores=scores)
    except ValueError as e:
        status = "400"
        metrics.inc_predict_requests(status)
        logger.warning("预测请求参数错误: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        status = "500"
        metrics.inc_predict_requests(status)
        logger.error("预测失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("启动 DeepFM 模型推理服务...")
    logger.info("模型路径: %s", DEEPFM_MODEL_PATH)
    logger.info("服务地址: http://%s:%s", HOST, PORT)
    logger.info("预测: http://%s:%s/predict 指标: http://%s:%s/metrics", HOST, PORT, HOST, PORT)

    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
