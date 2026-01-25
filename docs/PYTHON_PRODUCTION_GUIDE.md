# Python 生产工业级补充建议

本文档针对当前工程 `python/` 目录下的训练与推理服务，给出**工业级生产环境**的补充建议，便于从「可运行」升级到「可观测、可运维、可扩展」。

---

## 一、当前工程已具备的能力

| 能力 | 现状 |
|------|------|
| 训练 | XGBoost、DeepFM、Item2Vec；多数据源（file / OSS Parquet / MySQL） |
| 推理服务 | FastAPI HTTP；/predict、/health、/reload 热加载 |
| 自动化 | run_training.sh、evaluate、register_model、deploy；K8s CronJob |
| 容器化 | Dockerfile、docker-compose、健康检查 |
| 测试 | pytest、test_server、test_model_loader、test_integration |

以下建议在上述基础上查漏补缺，按**优先级**与**投入**分级。

---

## 二、可观测性（Observability）

### 2.1 结构化日志

**建议**：生产环境使用 **JSON 结构化日志**，便于 ELK/Loki 等采集与检索；附带 `request_id`、`model_version` 等上下文。

**可选实现**：

```python
# 使用 structlog 或 python-json-logger
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "model_version": getattr(record, "model_version", None),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)
```

- 在 FastAPI 中通过**中间件**为每个请求注入 `request_id`（如 `uuid4`），并绑定到 `contextvars`，在 predict / reload 日志中输出。

### 2.2 推理服务 Metrics（Prometheus）

**建议**：暴露 `/metrics` 端点，供 Prometheus 抓取。指标至少包括：

| 指标 | 类型 | 说明 |
|------|------|------|
| `http_requests_total` | Counter | 按 path、method、status 分桶 |
| `http_request_duration_seconds` | Histogram | 按 path 分桶，含 p50/p95/p99 |
| `model_predict_batch_size` | Histogram | 单次 /predict 的 batch 大小 |
| `model_reload_total` | Counter | /reload 调用次数 |
| `model_version_info` | Gauge | 当前加载的模型版本（label） |

**可选实现**：使用 `prometheus_client` + FastAPI 中间件（或 `starlette_exporter`）自动打点；`/predict` 内再打 batch_size、latency。

### 2.3 链路追踪（Tracing）

**建议**：接入 **OpenTelemetry**，为请求打 trace、span，便于排查跨服务延迟。

- HTTP 入口：FastAPI middleware 创建 root span。
- 调用模型预测：创建 child span，记录 `model_version`、`batch_size`。
- 与现有日志、Metrics 通过 `trace_id` / `request_id` 关联。

**优先级**：在已有日志 + Metrics 稳定后再上；多服务、多依赖时价值更大。

---

## 三、配置与密钥管理

### 3.1 配置集中管理

**建议**：严格 **12-Factor**，配置来自环境变量或配置文件，**禁止硬编码**。

- 模型路径、服务端口、超时等：环境变量（如 `MODEL_DIR`、`PORT`、`PREDICT_TIMEOUT`）。
- 多环境：`config/dev.yaml`、`config/prod.yaml` 或 env 前缀（`RECKIT_*`），由启动脚本选择。

### 3.2 密钥与敏感信息

**建议**：

- **不要**在代码或配置文件中写死 DB、OSS、API Key。
- 使用 **环境变量** 或 **密钥管理服务**（如 Vault、K8s Secret）注入。
- 当前 `data_loader`、`register_model`、`deploy` 已支持通过 env 传 OSS/MySQL 等，保持该模式即可。

---

## 四、安全

### 4.1 认证与鉴权

**建议**：生产环境对 **/predict**、**/reload** 做访问控制。

- **/predict**：API Key、JWT 或内部 service mesh 互信；避免公网裸暴露。
- **/reload**：仅允许发布系统或运维调用；可单独 API Key 或 IP 白名单。

**可选实现**：FastAPI `Depends()` 校验 `Authorization` 或 `X-API-Key`，失败返回 401。

### 4.2 输入校验与限流

**建议**：

- **请求体**：Pydantic 严格校验 `features_list` 长度、特征维度、数值范围；防止畸形请求或滥用。
- **限流**：按 IP 或 API Key 限制 QPS，防止打满推理服务。可接入 **slowapi**、**Kong** 等。

### 4.3 依赖与镜像安全

**建议**：

- 定期 `pip audit`、`safety check`，修复已知漏洞。
- 基础镜像使用带标签的官方镜像（如 `python:3.11-slim`），必要时做 CVE 扫描。

---

## 五、可靠性

### 5.1 就绪与存活探针分离

**建议**：K8s 将 **readiness** 与 **liveness** 分开。

- **liveness**：进程是否存活；可复用现有 `/health` 或简单 `GET /`。
- **readiness**：是否可以接收流量；**只有在模型成功加载**后才返回 200，否则 503。当前 `/health` 已区分「未加载」时 503，可继续沿用；若需更细粒度，可拆成 `/live`、`/ready`。

### 5.2 优雅退出（Graceful Shutdown）

**建议**：SIGTERM 时**优雅关闭**：停止接收新请求，等待正在进行的 /predict 完成后再退出。

**可选实现**：uvicorn 支持 `graceful_timeout`；在 FastAPI lifespan 或 signal handler 里标记 `shutting_down`，中间件对新建请求直接返回 503。

### 5.3 超时与重试

**建议**：

- **推理**：对单次 /predict 设置**服务端超时**（如 30s），防止慢请求堆积。
- **调用方**：Go 端 RPCModel 调用推理服务时，设置连接与读超时，并配置重试（如指数退避）。

---

## 六、CI/CD 与质量门禁

### 6.1 代码质量

**建议**：

- **Lint**：`ruff` 或 `flake8`；**格式化**：`black`。
- **类型检查**：`mypy` 对 `service/`、`train/` 跑一遍，逐步严苛。
- **pre-commit**：提交前自动跑 lint + format，减少脏代码入库。

### 6.2 测试与覆盖率

**建议**：

- **单元测试**：覆盖 `model_loader`、`data_loader`、关键工具函数；`pytest -v --cov=service --cov=train`。
- **集成测试**：起真实 FastAPI app，调 /health、/predict、/reload；可与 Docker 一起在 CI 中跑。
- **门禁**：CI 失败则禁止合入；覆盖率若未达目标（如 70%）仅告警，逐步提升。

### 6.3 构建与部署

**建议**：

- **镜像**：代码合入 main 或打 tag 时自动构建 Docker 镜像，推送至镜像仓库。
- **部署**： staging 自动部署；生产需审批或手动触发。当前已有 K8s 相关 yaml，可纳入 CI/CD 流水线。

---

## 七、训练与数据流水线

### 7.1 数据校验

**建议**：训练前对 **样本** 做基础校验，防止脏数据导致训练静默失败。

- **Pandera** 或 **Great Expectations**：校验列存在性、类型、范围、缺失率等。
- 与 `train/features.py`、`data/README.md` 约定一致，发现异常即失败并打日志。

### 7.2 实验与模型版本

**建议**：

- 每次训练**固定** `--version`（如日期+ Git short SHA），便于追溯。
- 训练日志、评估指标、模型路径写入 **MLflow** 或自建表，便于对比实验、回滚。

### 7.3 样本与特征可复现

**建议**：数据分区、随机种子、特征计算逻辑**可复现**；必要时对样本做 checksum，便于审计。

---

## 八、运维与预案

### 8.1 告警

**建议**：对以下情况告警（对接 PagerDuty、钉钉、企业微信等）：

- 推理服务 **5xx 率**、**延迟 p99** 超阈值。
- **/health** 连续失败（如 K8s 探针失败）。
- **模型 reload 失败**；或**当前模型版本**与预期不符。

### 8.2 故障排查与回滚

**建议**：

- **Runbook**：记录常见故障（如 OOM、模型未加载、依赖超时）及处理步骤。
- **回滚**：通过 `deploy.py` 切换回上一版本并触发 /reload；或 K8s 回滚到上一镜像。

### 8.3 容量与弹性

**建议**：

- 通过 **压测**（如 `locust`、`k6`）得到单实例 QPS、延迟，据此设定 HPA、资源 request/limit。
- 推理服务**无状态**，水平扩缩容即可；需注意模型加载带来的内存与冷启动时间。

---

## 九、建议落地优先级

| 优先级 | 项目 | 说明 |
|--------|------|------|
| **P0** | 结构化日志 + request_id | 排查问题的基础 |
| **P0** | /metrics + Prometheus | 监控、告警、SLO 依赖 |
| **P0** | /reload 鉴权 + 输入校验加强 | 安全与稳定性 |
| **P1** | 优雅退出 | 滚动发布、扩缩容时不丢请求 |
| **P1** | CI：lint + test + 镜像构建 | 质量与部署自动化 |
| **P1** | 训练前数据校验（Pandera 等） | 避免脏数据导致静默失败 |
| **P2** | Tracing（OpenTelemetry） | 多服务、复杂链路时再上 |
| **P2** | 限流、熔断 | 高 QPS、依赖多时考虑 |
| **P2** | MLflow / 实验管理 | 实验多、需对比时引入 |

---

## 十、与现有文档的关系

- **TRAINING_AUTOMATION**：偏重训练流程自动化、触发、编排；本文偏重 **推理服务** 与 **生产运维**。
- **DATA_LOADER**、**TRAINING_DATA**：数据源与格式；本文不重复，仅在「数据校验」等处引用。

按上述优先级逐步落地，可将当前 Python 工程提升到**工业级可运维、可观测、可扩展**的水平；若资源有限，优先完成 **P0** 再推进 **P1**。

---

## 十一、最小可落地示例

以下为**可直接复用**的最小实现，便于快速接入 request_id、/metrics、优雅退出。

### 11.1 Request ID 中间件

```python
# service/middleware_request_id.py（可选）
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_CTX_KEY = "request_id"
request_id_ctx: ContextVar[str] = ContextVar(REQUEST_ID_CTX_KEY, default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_ctx.reset(token)
```

在 FastAPI 中挂载：`app.add_middleware(RequestIDMiddleware)`。日志中可通过 `request_id_ctx.get()` 输出 request_id。

### 11.2 Prometheus /metrics 端点

```python
# 依赖: pip install prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

PREDICT_REQUESTS = Counter("model_predict_requests_total", "Total /predict requests", ["status"])
PREDICT_LATENCY = Histogram("model_predict_duration_seconds", "Predict latency")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 在 /predict 内: 用 PREDICT_LATENCY.time() 包一层，按 status 打 PREDICT_REQUESTS
```

### 11.3 优雅退出（uvicorn）

```bash
# 启动时指定 graceful_timeout，SIGTERM 后等待 in-flight 请求完成再退出
uvicorn service.server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30
# K8s 需配合 lifecycle.preStop 睡眠几秒，让 ingress 先摘除后端
```

K8s Pod 的 `terminationGracePeriodSeconds` 建议 ≥ 30，`preStop` 可 `sleep 5` 再退出，便于负载均衡摘除。

---

将上述中间件与 /metrics 接入现有 `service/server.py`、`deepfm_server.py`，即可在不大改结构的前提下提升可观测性与可靠性；完整改造可参考本文第二～五节。
