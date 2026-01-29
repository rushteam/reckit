# Python 模型服务

推理服务入口与领域层，与 Reckit 协议对齐。

## 协议约束

**以 Reckit 为准**：路径、请求/响应格式、健康检查见 Reckit 仓库 [docs/MODEL_SERVICE_PROTOCOL.md](../../docs/MODEL_SERVICE_PROTOCOL.md)。  
本目录 `domain/protocol.py` 中的 `TorchServePredictRequest` / `TorchServePredictResponse` 与该约束一致。

## 目录与使用

| 目录/文件 | 说明 |
|-----------|------|
| `domain/` | 协议约定（TorchServe 请求/响应） |
| `app/` | 用例（如批量预测） |
| `server.py` | XGBoost 推理服务入口 |
| `deepfm_server.py` | DeepFM 推理服务入口 |
| `mmoe_server.py` | MMoE 推理服务入口 |
| `two_tower_server.py` | 双塔推理服务入口 |
| `unified_server.py` | **统一推理服务**：单进程多模型，按 model_name 分发 |
| `*_model_loader.py` | 各模型加载与推理实现 |

**统一服务（推荐）**

单进程暴露所有启用的模型，通过 `POST /predictions/{model_name}` 按模型名分发；双塔使用 `model_name=user_tower` 与 `model_name=item_tower` 区分。

```bash
# 启用多个模型（逗号分隔），默认仅 xgb
ENABLED_MODELS=xgb,deepfm,mmoe,user_tower,item_tower,youtube_dnn,dssm,graph_recall \
  uvicorn service.unified_server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30
```

**启动示例（单模型，按需使用）**

```bash
# XGBoost（默认 8080）
uvicorn service.server:app --host 0.0.0.0 --port 8080

# DeepFM
uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080

# MMoE（建议 8081）
uvicorn service.mmoe_server:app --host 0.0.0.0 --port 8081

# 双塔（建议 8085）
uvicorn service.two_tower_server:app --host 0.0.0.0 --port 8085
```

**Go 调用**：使用 `model.NewRPCModel(..., "http://host:port/predictions/xgb", ...)`、`service.NewTorchServeClient("http://host:port", "user_tower", ...)`（双塔用 `user_tower`/`item_tower`）、`rerank.MMoENode{ Endpoint: "http://host:port/predictions/mmoe", ... }` 等，见 Reckit [docs/MODEL_SERVICE_PROTOCOL.md](../../docs/MODEL_SERVICE_PROTOCOL.md)。
