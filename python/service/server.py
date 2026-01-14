#!/usr/bin/env python3
"""
XGBoost 模型推理服务

使用 FastAPI 实现 HTTP 服务，与 Go 端的 RPCModel 协议对齐。

启动方式:
    uvicorn service.server:app --host 0.0.0.0 --port 8080

或者:
    python service.server:app
"""
import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入模型加载器（使用相对导入）
import importlib.util
loader_path = os.path.join(os.path.dirname(__file__), "model_loader.py")
spec = importlib.util.spec_from_file_location("model_loader", loader_path)
model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)
ModelLoader = model_loader_module.ModelLoader

# 模型路径（相对于项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")

# 从环境变量读取配置
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")

# 创建 FastAPI 应用
app = FastAPI(
    title="Reckit XGBoost Model Service",
    description="XGBoost 模型推理服务，与 Go 端 RPCModel 协议对齐",
    version="1.0.0",
)

# 全局模型加载器
model_loader = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_loader
    try:
        logger.info("正在启动模型服务...")
        logger.info(f"模型路径: {MODEL_PATH}")
        logger.info(f"特征元数据路径: {FEATURE_META_PATH}")
        
        model_loader = ModelLoader(MODEL_PATH, FEATURE_META_PATH, MODEL_VERSION)
        model_loader.load()
        
        logger.info("模型服务启动成功！")
        logger.info(f"模型版本: {model_loader.model_version or 'unknown'}")
        logger.info(f"特征数量: {model_loader.feature_count}")
    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        logger.error("请先运行训练脚本: python train/train_xgb.py")
        raise
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        raise


# 请求/响应模型（与 Go 端协议对齐）
class PredictRequest(BaseModel):
    """批量预测请求（与 Go RPCModel 协议对齐）"""
    features_list: list[dict[str, float]]  # 特征字典列表，例如 [{"ctr": 0.15, "cvr": 0.08, ...}, ...]


class PredictResponse(BaseModel):
    """批量预测响应（与 Go RPCModel 协议对齐）"""
    scores: list[float]  # 预测分数列表


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Reckit XGBoost Model Service",
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


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    批量预测接口（与 Go RPCModel 协议对齐）
    
    请求格式:
        {
            "features_list": [
                {
                    "ctr": 0.15,
                    "cvr": 0.08,
                    "price": 99.0,
                    "age": 25.0,
                    "gender": 1.0,
                    ...
                },
                ...
            ]
        }
    
    响应格式:
        {
            "scores": [0.85, 0.72, ...]
        }
    """
    if model_loader is None:
        logger.error("模型未加载，无法进行预测")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.debug(f"收到批量预测请求，样本数: {len(request.features_list)}")
        
        # 调用模型批量预测
        scores = model_loader.predict(request.features_list)
        
        logger.debug(f"批量预测完成，返回分数数量: {len(scores)}")
        return PredictResponse(scores=scores)
    except ValueError as e:
        logger.warning(f"预测请求参数错误: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动 XGBoost 模型推理服务...")
    logger.info(f"模型路径: {MODEL_PATH}")
    logger.info(f"服务地址: http://{HOST}:{PORT}")
    logger.info(f"预测接口: http://{HOST}:{PORT}/predict")
    
    uvicorn.run(app, host=HOST, port=PORT)
