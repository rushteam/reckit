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
| `*_model_loader.py` | 各模型加载与推理实现 |

**启动示例**

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

**Go 调用**：使用 `model.NewRPCModel(..., "http://host:port/predictions/xgb", ...)`、`service.NewTorchServeClient("http://host:port", "two_tower", ...)`、`rerank.MMoENode{ Endpoint: "http://host:port/predictions/mmoe", ... }` 等，见 Reckit [docs/MODEL_SERVICE_PROTOCOL.md](../../docs/MODEL_SERVICE_PROTOCOL.md)。
