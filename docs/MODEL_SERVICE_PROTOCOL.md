# 模型服务协议约束（Python 服务参考）

Reckit 为一等公民：**协议以本仓库约定为准**，Python 模型服务需按此实现。Reckit 约定尽量参考 [TorchServe Inference API](https://docs.pytorch.org/serve/inference_api.html)、[KServe v2 handler 约束](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/) 等标准。

---

## 一、传输与框架概念区分

| 概念 | 说明 |
|------|------|
| **TorchServe 协议（传输）** | 路径 `POST /predictions/{model_name}`，请求 `{"data": [...]}`，响应 `{"predictions": [...]}`，健康 `GET /ping` → `{"status": "Healthy"}`。凡符合此约定的服务，Go 端均通过 **TorchServeClient** 调用。 |
| **TorchServeClient** | Go 端唯一对接 TorchServe 协议的客户端（`service.TorchServeClient`）。Rank、Rerank、双塔等凡走 TorchServe 协议的服务，底层均应由 TorchServeClient 发请求。 |
| **框架接口（业务）** | Pipeline 使用的抽象：`model.RankModel`（排序）、`core.MLService`（推理）、`pipeline.Node`（Rerank/Recall）。实现层用 TorchServeClient 或专用 HTTP 满足协议。 |

- **语义一致**（特征列表 → 分数/向量/对象列表）：必须走 TorchServe 协议，Go 端用 TorchServeClient（或封装 TorchServeClient 的 RPCModel/MMoENode）。
- **语义不一致**（如单条 query embedding、user_id+top_k 召回）：保持专用路径与 body，Go 端用专用 HTTP 客户端。

---

## 二、协议约束表（Python 服务必遵）

以下为 Reckit 对 Python 模型服务的约束；Python 实现需与此表一致。

### 1. TorchServe 协议（语义一致服务）

| 项目 | 约束 |
|------|------|
| 路径 | `POST /predictions/{model_name}` |
| 请求 | `{"data": [{"feature_a": 0.1, ...}, ...]}` |
| 响应 | `{"predictions": [...]}`（标量列表 / embedding 展平 / 对象列表如 `[{ctr, watch_time, gmv}, ...]`） |
| 健康 | `GET /ping` → 200，body `{"status": "Healthy"}` |

| 服务类型 | 建议 model_name | 响应 predictions 含义 |
|----------|-----------------|------------------------|
| 排序（Rank） | xgb / deepfm | 标量分列表，与 data 条数一致 |
| 重排（Rerank） | mmoe | 对象列表，每项 `{ctr, watch_time, gmv}` |
| 双塔/Embedding | two_tower | 单条时为 User Embedding 向量（展平 float 列表） |

Go 端用法：

- **Rank**：`model.RPCModel`（内部发 TorchServe 协议，Endpoint 填 `http://host:port/predictions/xgb` 等）。
- **Rerank**：`rerank.MMoENode`（同协议，Endpoint 填 `http://host:port/predictions/mmoe`）。
- **双塔 / MLService**：`service.NewTorchServeClient("http://host:port", "two_tower", ...)`。

### 2. 专用协议（语义不一致服务）

| 服务 | 路径 | 请求 | 响应 |
|------|------|------|------|
| DSSM | `POST /query_embedding` | `{"query_features": {...}}` | `{"query_embedding": [...]}` |
| YouTube DNN | `POST /user_embedding` | `{"user_features": {...}, "history_item_ids": [...]}` | `{"user_embedding": [...]}` |
| Graph 召回 | `POST /recall` | `{"user_id": "...", "top_k": N}` | `{"item_ids": [...], "scores": [...]}` |

上述服务仍须提供 `GET /ping` → `{"status": "Healthy"}`。Go 端由各 Recall 实现内自行发 HTTP，不通过 TorchServeClient。

---

## 三、小结

- **传输统一**：凡「data → predictions」语义的服务，一律 TorchServe 协议，Go 端一律 TorchServeClient（或基于 TorchServeClient 的适配器）。
- **框架接口**：RankModel、MLService、ReRank/Recall Node 为业务抽象；实现层按上表选 TorchServeClient 或专用 HTTP。
- **Python 侧**：以本表为约束实现路径与 body，可参考 `python/service/domain/protocol.py` 中的请求/响应模型。
