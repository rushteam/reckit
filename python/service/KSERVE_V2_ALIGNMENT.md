# KServe v2（Open Inference Protocol）约束参考

在需要约定**自定义 handler 的请求/响应结构**时，尽量采用 [KServe v2 / Open Inference Protocol](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/) 的**数据约束**（不新增 v2 路由）。

- **不增加 v2 路由**：本仓库不提供 `GET /v2/health/ready`、`GET /v2/models/{model_name}`、`POST /v2/models/{model_name}/infer` 等路径。
- **仅约束 handler 约定**：推理请求/响应的**语义**（输入张量名、shape、datatype、输出结构）可参考 v2，在现有 `POST /predictions/{model_name}` 的 body 中体现。

参考：
- [Open Inference Protocol (V2)](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/)
- [open_inference_rest.yaml](https://raw.githubusercontent.com/kserve/open-inference-protocol/main/specification/protocol/open_inference_rest.yaml)

---

## 一、KServe v2 约束摘要（仅作 handler 约定参考）

### 1. 推理请求（Inference Request）结构

```json
{
  "id": $string,           // optional
  "parameters": $object,   // optional
  "inputs": [
    {
      "name": $string,
      "shape": [ $number, ... ],
      "datatype": $string,   // e.g. "FP64", "FP32"
      "parameters": $object, // optional
      "data": $tensor_data   // array of numbers (row-major)
    }
  ],
  "outputs": [ { "name": $string, "parameters": $object } ]  // optional
}
```

- 输入为**张量**：`name`、`shape`、`datatype`、`data`（一维或嵌套数组，行优先）。
- 多输入时每个元素一个张量；单输入时通常一个 `inputs` 元素，如 `"features"`。

### 2. 推理响应（Inference Response）结构

```json
{
  "model_name": $string,
  "model_version": $string,  // optional
  "id": $string,             // optional, echo request id
  "parameters": $object,     // optional
  "outputs": [
    {
      "name": $string,
      "shape": [ $number, ... ],
      "datatype": $string,
      "parameters": $object,  // optional
      "data": $tensor_data
    }
  ]
}
```

- 输出为**张量**：`name`、`shape`、`datatype`、`data`。

---

## 二、本仓库 handler 约定与 v2 的对应（仅约束，无 v2 路由）

- **路径**：保持 `POST /predictions/{model_name}`、`GET /ping`（TorchServe 风格），**不**新增 `/v2/*` 路由。
- **请求**：当前使用 `{"data": [{"f1": 0.1, "f2": 0.2}, ...]}`；若扩展为张量格式，可参考 v2 的 `inputs`（name/shape/datatype/data），特征顺序与模型 `feature_columns` 一致。
- **响应**：当前使用 `{"predictions": [0.85, ...]}`；若扩展为张量格式，可参考 v2 的 `outputs`（name/shape/datatype/data）。
- **健康检查**：`GET /ping` 返回 `{ "status": "Healthy" }`，与 TorchServe 一致；不新增 v2 健康路径。

### 各模型 handler 语义（与 v2 张量约定对齐）

| 模型 | 输入语义 | 输出语义 |
|------|----------|----------|
| XGB / DeepFM | 等价 `features`: shape `[batch, n_features]`, datatype `FP64` | 等价 `predictions`: shape `[batch]`, datatype `FP64` |
| MMoE | 同上 | 等价 `predictions`: shape `[batch, 3]`（ctr/watch_time/gmv） |
| Two Tower | 等价 `features`: shape `[batch, n_user_features]`, datatype `FP64` | 等价 `predictions`: shape `[batch, embed_dim]`, datatype `FP64` |

特征顺序以各服务/模型元数据（如 `feature_columns`）为准。协议约束以 Reckit 仓库 [docs/MODEL_SERVICE_PROTOCOL.md](../../docs/MODEL_SERVICE_PROTOCOL.md) 为准。
