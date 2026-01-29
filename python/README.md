# Python ML 训练与服务

本目录包含机器学习模型的训练和推理服务代码。**协议以 Reckit 为准**，见仓库 [docs/MODEL_SERVICE_PROTOCOL.md](../docs/MODEL_SERVICE_PROTOCOL.md)。

**支持的模型**：
- **XGBoost**：树模型，训练快、推理快（`train/train_xgb.py` + `service/server.py`）
- **DeepFM**：PyTorch 深度学习模型，自动学习特征交互（`train/train_deepfm.py` + `service/deepfm_server.py`）
- **MMoE**：多目标重排（CTR / 时长 / GMV）（`train/train_mmoe.py` + `service/mmoe_server.py`）
- **YouTube DNN**：视频/内容流召回用户向量（`train/train_youtube_dnn.py` + `service/youtube_dnn_server.py`）
- **DSSM**：Query-Doc 语义召回（`train/train_dssm.py` + `service/dssm_server.py`）
- **Node2Vec**：图嵌入、社交/关注页召回（`train/train_node2vec.py` + `service/graph_recall_server.py`）
- **Word2Vec / Item2Vec**：文本/序列向量化（`train/train_item2vec.py`）

## 目录结构

```
python/
├── data/              # 训练数据（CSV / TXT）
│   └── README.md      # 示例数据格式说明（train_data、behavior、corpus）
├── train/            # 训练脚本
│   ├── train_xgb.py  # XGBoost 训练脚本
│   ├── train_deepfm.py  # DeepFM PyTorch 训练脚本
│   ├── train_item2vec.py  # Word2Vec / Item2Vec 训练脚本
│   ├── features.py   # 特征配置
│   └── DEEPFM_README.md  # DeepFM 使用说明
├── service/          # 推理服务
│   ├── domain/       # 协议约定（TorchServe 请求/响应）
│   ├── app/          # 用例（批量预测等）
│   ├── server.py     # XGBoost 推理入口
│   ├── unified_server.py  # 统一推理入口（单进程多模型，按 model_name 分发；双塔用 user_tower/item_tower）
│   ├── deepfm_server.py, mmoe_server.py, two_tower_server.py, ...
│   ├── *_model_loader.py  # 各模型加载与推理
│   └── README.md     # 使用说明（协议约束见 reckit docs）
├── scripts/          # 训练自动化脚本
│   ├── run_training.sh  # 训练流水线主脚本
│   ├── evaluate.py      # 模型评估与门控
│   ├── register_model.py  # 模型注册（上传到 S3/OSS）
│   ├── deploy.py        # 模型部署（触发 reload）
│   └── README.md         # 脚本使用说明
├── model/            # 训练好的模型（自动生成）
│   ├── xgb_model.json      # XGBoost 模型文件
│   ├── deepfm_model.pt    # DeepFM PyTorch 模型文件
│   ├── feature_meta.json   # XGBoost 特征元数据
│   ├── deepfm_feature_meta.json  # DeepFM 特征元数据
│   └── feature_scaler.json # 特征标准化器（可选）
├── tests/            # 测试文件
│   ├── test_model_loader.py  # 模型加载器测试
│   ├── test_server.py        # 服务器测试
│   └── test_integration.py   # 集成测试
├── pyproject.toml    # 项目与依赖（uv）
├── uv.lock           # 锁文件（可选，本地运行 uv lock 生成）
├── requirements.txt  # 已迁移至 pyproject.toml，仅作参考
├── Dockerfile        # Docker 镜像（uv + 统一服务）
├── docker-compose.yml # Docker Compose（统一服务）
└── README.md         # 本文件
```

## 快速开始

### 1. 安装依赖（uv）

本目录使用 [uv](https://docs.astral.sh/uv/) 管理依赖，依赖声明在 `pyproject.toml`。

```bash
cd python
# 安装 uv：https://docs.astral.sh/uv/getting-started/installation/
uv sync
# 可选：生成锁文件便于复现
uv lock
```

若未使用 uv，可沿用 `pip install -r requirements.txt`（与 pyproject.toml 内容一致，仅作备用）。

### 2. 训练模型

#### XGBoost 模型

```bash
# 基础训练
python train/train_xgb.py

# 指定模型版本
python train/train_xgb.py --version v1.0.0

# 启用特征标准化
python train/train_xgb.py --normalize
```

#### DeepFM 模型（PyTorch）

```bash
# 基础训练
python train/train_deepfm.py

# 指定模型版本和训练参数
python train/train_deepfm.py --version v1.0.0 --epochs 100 --batch-size 64
```

**详细说明**：见 `train/DEEPFM_README.md`

#### MMoE / YouTube DNN / DSSM / Node2Vec

```bash
# MMoE 多目标（CTR、watch_time、gmv）
python train/train_mmoe.py [--data-path data/mmoe_train_data.csv] [--epochs 50]

# YouTube DNN（用户特征 + 历史 -> 用户向量）
python train/train_youtube_dnn.py [--data data/youtube_dnn_data.csv] [--epochs 20]

# DSSM（Query-Doc 语义匹配）
python train/train_dssm.py [--data data/dssm_data.csv] [--epochs 20]

# Node2Vec（边表 -> 节点嵌入，社交/关注页召回）
python train/train_node2vec.py [--edges data/graph_edges.csv] [--dim 64] [--epochs 10]
```

对应 Golang：`rerank.MMoENode`、`recall.YouTubeDNNRecall`、`recall.DSSMRecall`、`recall.GraphRecall`。详见 `docs/MODEL_COVERAGE_ANALYSIS.md`。

#### 数据源：文件 / OSS Parquet / MySQL 协议

训练脚本支持多种数据源（`--data-source file|oss|mysql|doris`）：

```bash
# 本地 CSV（默认）
python train/train_xgb.py --data-source file --data-path data/train_data.csv

# OSS Parquet（S3、阿里云 OSS、腾讯云 COS、MinIO）
python train/train_xgb.py --data-source oss --data-path s3://bucket/train.parquet
python train/train_xgb.py --data-source oss --data-path oss://bucket/train.parquet \
  --oss-endpoint https://oss-cn-hangzhou.aliyuncs.com

# MySQL 协议（MySQL / Doris / TiDB）
python train/train_xgb.py --data-source mysql --doris-query "SELECT * FROM db.table" \
  --doris-host 127.0.0.1 --doris-port 3306 --doris-user root --doris-password xxx

# Doris (同 mysql)
python train/train_xgb.py --data-source doris --doris-query "SELECT * FROM db.table" \
  --doris-host 127.0.0.1 --doris-port 9030 --doris-user root --doris-password xxx
```

**依赖**：OSS 需 `pyarrow`、`s3fs`；MySQL 协议需 `pymysql`。详见 `train/DATA_LOADER.md`。

### 3. 启动推理服务

#### 统一服务（推荐）

单进程多模型，按 `model_name` 分发；双塔用 `user_tower` / `item_tower`。

```bash
ENABLED_MODELS=xgb,deepfm,mmoe,user_tower,item_tower,youtube_dnn,dssm,graph_recall \
  uvicorn service.unified_server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30
```

Go 端统一访问同一端口，例如：`/predictions/xgb`、`/predictions/deepfm`、`/predictions/user_tower`、`/predictions/item_tower`、`/predictions/mmoe` 等。

#### XGBoost 服务（单模型）

```bash
# 方式 1: 使用 uvicorn（推荐加 --timeout-keep-alive 30 便于优雅退出）
uvicorn service.server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30

# 方式 2: 直接运行
python -m service.server
```

#### DeepFM 服务（单模型）

```bash
uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 30
python -m service.deepfm_server
```

服务启动后，可以通过以下方式测试：

```bash
# 健康检查
curl http://localhost:8080/health

# Prometheus 指标
curl http://localhost:8080/metrics

# 模型热加载（reload，无需重启服务）
curl -X POST http://localhost:8080/reload

# 若配置了 RELOAD_API_KEY，需带 X-API-Key 头
curl -X POST http://localhost:8080/reload -H "X-API-Key: your-secret"
```

请求会自动带上 `X-Request-ID`（若客户端未传则服务端生成），便于日志与链路关联。

# 批量预测接口（特征名带前缀，与 FEATURE_COLUMNS 对齐）
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      {
        "item_ctr": 0.15,
        "item_cvr": 0.08,
        "item_price": 99.0,
        "user_age": 25.0,
        "user_gender": 1.0,
        "cross_age_x_ctr": 3.75,
        "cross_gender_x_price": 99.0
      }
    ]
  }'
```

### 4. 在 Go 端使用

在 Go 代码中使用 `RPCModel` 调用 Python 服务：

```go
xgbModel := model.NewRPCModel("xgboost", "http://localhost:8080/predict", 5*time.Second)
rpcNode := &rank.RPCNode{Model: xgbModel}
```

## 协议说明

### 请求格式

与 Go 端 `RPCModel.Predict` 协议对齐（特征名带前缀，与 FEATURE_COLUMNS 对齐）：

```json
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
    }
  ]
}
```

**注意**: 
- `user_gender` 值为 0=未知，1=男，2=女
- 特征名必须与 `feature_meta.json` 中的 `feature_columns` 完全一致

### 响应格式

```json
{
  "scores": [0.85]
}
```

## 特征配置

特征列定义在 `train/features.py` 中（带前缀，与 EnrichNode / RPCNode 默认不 strip 对齐）：

- `item_ctr`: 物品点击率
- `item_cvr`: 物品转化率
- `item_price`: 物品价格
- `user_age`: 用户年龄
- `user_gender`: 用户性别（0=未知，1=男，2=女）
- `cross_age_x_ctr`: 年龄 × CTR 交叉特征
- `cross_gender_x_price`: 性别 × 价格交叉特征

**注意**：特征名必须与 `feature_meta.json` 中的 `feature_columns` 完全一致。

## 扩展指南

### 使用自己的数据

1. 准备 CSV 数据文件，包含 `train/features.py` 中定义的特征列
2. 将数据文件放到 `data/` 目录
3. 修改 `train/train_xgb.py` 中的数据路径
4. 重新训练模型

各示例数据（`train_data.csv`、`behavior.csv`、`corpus.txt`）的格式与生成规则见 **`data/README.md`**。

### 添加新特征

1. 在 `train/features.py` 的 `FEATURE_COLUMNS` 中添加新特征名
2. 确保训练数据包含该特征列
3. 重新训练模型
4. 在 Go 端的特征注入节点中生成该特征

### 支持其他算法

可以类似地实现 LightGBM、CatBoost 等：

1. 修改 `train/train_xgb.py` 使用其他算法
2. 修改 `service/model_loader.py` 加载其他格式的模型
3. 保持 HTTP 协议不变（与 Go 端对齐）

## 测试

运行测试套件：

```bash
# 运行所有测试
python -m unittest discover -s tests -p "test_*.py" -v

# 或使用测试脚本
bash tests/run_tests.sh

# 使用 pytest（如果安装了）
pytest tests/ -v
```

## Docker 部署

镜像使用 **uv** 安装依赖，默认启动 **统一推理服务**（`service.unified_server`），通过 `POST /predictions/{model_name}` 按模型名分发。

### 构建镜像

```bash
docker build -t reckit-unified-service .
```

### 使用 Docker Compose

```bash
# 启动统一服务（默认仅启用 xgb）
docker-compose up -d

# 启用多个模型
ENABLED_MODELS=xgb,deepfm,mmoe,user_tower,item_tower docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 环境变量

- `PORT`: 服务端口（默认: 8080）
- `HOST`: 服务地址（默认: 0.0.0.0）
- `ENABLED_MODELS`: 启用的模型，逗号分隔（默认: xgb）。可选：xgb, deepfm, mmoe, user_tower, item_tower, youtube_dnn, dssm, graph_recall
- `MODEL_VERSION`: 模型版本（可选）

## 新功能特性

### ✅ 模型版本管理

训练时自动生成版本号（时间戳），或通过 `--version` 参数指定：

```bash
python train/train_xgb.py --version v1.0.0
```

模型版本信息保存在 `feature_meta.json` 中，服务启动时会显示。

### ✅ 特征标准化

支持训练时进行特征标准化，推理时自动应用：

```bash
python train/train_xgb.py --normalize
```

标准化参数保存在 `feature_scaler.json` 中。

### ✅ 特征验证

- 自动检测缺失特征（使用默认值 0.0 并记录警告）
- 验证特征类型和有效性（NaN/Inf 处理）
- 详细的日志记录

### ✅ 完善的日志系统

使用 Python `logging` 模块，支持：
- 不同日志级别（INFO, WARNING, ERROR, DEBUG）
- 结构化日志输出
- 异常堆栈跟踪

### ✅ 错误处理

- 优雅的错误处理和用户友好的错误消息
- HTTP 状态码正确映射（400, 500, 503）
- 详细的错误日志

### ✅ 单元测试和集成测试

- `test_model_loader.py`: 模型加载器单元测试
- `test_server.py`: 服务器接口测试
- `test_integration.py`: 端到端集成测试

### ✅ 模型热加载（Reload）

推理服务支持通过 `/reload` 端点热加载新模型，无需重启服务：

```bash
# 触发模型热加载
curl -X POST http://localhost:8080/reload
```

**特性**:
- 线程安全：使用锁保证 reload 期间预测请求等待
- 原子性替换：新模型加载成功后才替换旧模型
- 版本追踪：返回旧版本和新版本信息
- 错误处理：reload 失败不影响现有服务

**使用场景**:
- 模型版本更新
- 模型文件更新（通过 PVC 或文件同步）
- 自动化部署流程

### ✅ 训练流程自动化

提供完整的训练自动化脚本（`scripts/` 目录）：

- **run_training.sh**: 训练流水线主脚本（数据生成 → 训练 → 评估 → 注册）
- **evaluate.py**: 模型评估与门控（判断新模型是否优于当前模型）
- **register_model.py**: 模型注册（打包上传到 S3/OSS，本地记录版本）
- **deploy.py**: 模型部署（拉取模型，触发 reload）

详见 `scripts/README.md`。

### ✅ Kubernetes 部署

提供 K8s 配置文件（`k8s/` 目录）：

- **training-cronjob.yaml**: 训练任务 CronJob（每日自动执行）
- **inference-deployment.yaml**: 推理服务 Deployment（支持模型热加载）

详见 `k8s/README.md`。

## 生产环境建议

- ✅ 使用 Docker 容器化部署（已实现）
- ✅ 添加模型版本管理（已实现）
- ✅ 实现特征标准化 Pipeline（已实现）
- ✅ 添加监控和日志（已实现）
- ✅ 实现模型热更新（已实现，通过 `/reload` 端点）
- ✅ 训练流程自动化（已实现，`scripts/` 目录）
- ✅ Kubernetes 部署配置（已实现，`k8s/` 目录）
- ⏳ 使用 gRPC 替代 HTTP（性能更好，待实现）

**工业级补充建议**：可观测性（结构化日志、request_id、Prometheus /metrics）、安全（/reload 鉴权、输入校验、限流）、可靠性（优雅退出、readiness/liveness 分离）、CI/CD（lint、test、镜像构建）等，详见 [docs/PYTHON_PRODUCTION_GUIDE.md](../docs/PYTHON_PRODUCTION_GUIDE.md)。
