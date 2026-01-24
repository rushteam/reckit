#!/usr/bin/env python3
"""
DeepFM 模型推理服务

使用 FastAPI 实现 HTTP 服务，与 Go 端的 RPCModel 协议对齐。

启动方式:
    uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080

或者:
    python service.deepfm_server:app
"""
import logging
import os
import sys
import threading
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

# 导入模型加载器
import importlib.util
loader_path = os.path.join(os.path.dirname(__file__), "deepfm_model_loader.py")
spec = importlib.util.spec_from_file_location("deepfm_model_loader", loader_path)
model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)
DeepFMModelLoader = model_loader_module.DeepFMModelLoader

# 模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEEPFM_MODEL_PATH = os.path.join(MODEL_DIR, "deepfm_model.pt")
DEEPFM_FEATURE_META_PATH = os.path.join(MODEL_DIR, "deepfm_feature_meta.json")

# 从环境变量读取配置
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")

# 创建 FastAPI 应用
app = FastAPI(
    title="Reckit DeepFM Model Service",
    description="DeepFM 模型推理服务，与 Go 端 RPCModel 协议对齐",
    version="1.0.0",
)

# 全局模型加载器
model_loader = None
# 模型加载锁（保证 reload 时线程安全）
model_lock = threading.RLock()


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_loader
    try:
        logger.info("正在启动 DeepFM 模型服务...")
        logger.info(f"模型路径: {DEEPFM_MODEL_PATH}")
        logger.info(f"特征元数据路径: {DEEPFM_FEATURE_META_PATH}")
        
        model_loader = DeepFMModelLoader(DEEPFM_MODEL_PATH, DEEPFM_FEATURE_META_PATH, MODEL_VERSION)
        model_loader.load()
        
        logger.info("DeepFM 模型服务启动成功！")
        logger.info(f"模型版本: {model_loader.model_version or 'unknown'}")
        logger.info(f"特征数量: {model_loader.feature_count}")
    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        logger.error("请先运行训练脚本: python train/train_deepfm.py")
        raise
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        raise


# 请求/响应模型（与 Go 端协议对齐）
class PredictRequest(BaseModel):
    """批量预测请求（与 Go RPCModel 协议对齐）"""
    features_list: list[dict[str, float]]  # 特征字典列表


class PredictResponse(BaseModel):
    """批量预测响应（与 Go RPCModel 协议对齐）"""
    scores: list[float]  # 预测分数列表


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


@app.post("/reload")
async def reload_model():
    """
    重新加载模型（支持热更新）
    
    从 MODEL_DIR 目录重新加载模型和特征元数据，支持模型版本更新。
    使用锁保证线程安全，reload 期间预测请求会等待。
    """
    global model_loader
    
    with model_lock:
        try:
            logger.info("开始重新加载 DeepFM 模型...")
            logger.info(f"模型路径: {DEEPFM_MODEL_PATH}")
            logger.info(f"特征元数据路径: {DEEPFM_FEATURE_META_PATH}")
            
            # 创建新的模型加载器
            new_loader = DeepFMModelLoader(DEEPFM_MODEL_PATH, DEEPFM_FEATURE_META_PATH, MODEL_VERSION)
            new_loader.load()
            
            # 原子性替换
            old_version = model_loader.model_version if model_loader else None
            model_loader = new_loader
            
            logger.info(f"DeepFM 模型重新加载成功！")
            logger.info(f"旧版本: {old_version or 'unknown'}")
            logger.info(f"新版本: {model_loader.model_version or 'unknown'}")
            logger.info(f"特征数量: {model_loader.feature_count}")
            
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "old_version": old_version,
                "new_version": model_loader.model_version,
                "feature_count": model_loader.feature_count,
            }
        except FileNotFoundError as e:
            logger.error(f"模型文件未找到: {e}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {str(e)}")
        except Exception as e:
            logger.error(f"模型重新加载失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    批量预测接口（与 Go RPCModel 协议对齐）
    
    请求格式:
        {
            "features_list": [
                {
                    "item_ctr": 0.15,
                    "item_cvr": 0.08,
                    "item_price": 99.0,
                    "user_age": 25.0,
                    "user_gender": 1.0,
                    "cross_age_x_ctr": 3.75,
                    "cross_gender_x_price": 99.0
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
    
    # 使用锁保证 reload 期间预测请求等待
    with model_lock:
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
    
    logger.info("启动 DeepFM 模型推理服务...")
    logger.info(f"模型路径: {DEEPFM_MODEL_PATH}")
    logger.info(f"服务地址: http://{HOST}:{PORT}")
    logger.info(f"预测接口: http://{HOST}:{PORT}/predict")
    
    uvicorn.run(app, host=HOST, port=PORT)
