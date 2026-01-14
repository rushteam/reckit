# Python ML 训练与服务

本目录包含 XGBoost 模型的训练和推理服务代码，与 Go 端的 `RPCModel` 协议对齐，实现端到端的推荐系统闭环。

## 目录结构

```
python/
├── data/              # 训练数据（CSV）
├── train/            # 训练脚本
│   ├── train_xgb.py  # XGBoost 训练脚本
│   └── features.py   # 特征配置
├── service/          # 推理服务
│   ├── server.py     # FastAPI HTTP 服务
│   └── model_loader.py  # 模型加载器
├── model/            # 训练好的模型（自动生成）
│   ├── xgb_model.json      # XGBoost 模型文件
│   ├── feature_meta.json   # 特征元数据
│   └── feature_scaler.json # 特征标准化器（可选）
├── tests/            # 测试文件
│   ├── test_model_loader.py  # 模型加载器测试
│   ├── test_server.py        # 服务器测试
│   └── test_integration.py   # 集成测试
├── requirements.txt  # Python 依赖
├── Dockerfile        # Docker 镜像构建文件
├── docker-compose.yml # Docker Compose 配置
└── README.md         # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd python
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练
python train/train_xgb.py

# 指定模型版本
python train/train_xgb.py --version v1.0.0

# 启用特征标准化
python train/train_xgb.py --normalize

# 组合使用
python train/train_xgb.py --version v1.0.0 --normalize
```

这会：
- 生成示例训练数据（如果 `data/train_data.csv` 不存在）
- 训练 XGBoost 模型
- 保存模型到 `model/xgb_model.json`
- 保存特征元数据到 `model/feature_meta.json`（包含模型版本）
- 如果使用 `--normalize`，还会保存特征标准化器到 `model/feature_scaler.json`

### 3. 启动推理服务

```bash
# 方式 1: 使用 uvicorn
uvicorn service.server:app --host 0.0.0.0 --port 8080

# 方式 2: 直接运行
python service/server.py

# 方式 3: 使用 Docker
docker-compose up -d

# 方式 4: 使用环境变量配置
MODEL_VERSION=v1.0.0 PORT=8080 python service/server.py
```

服务启动后，可以通过以下方式测试：

```bash
# 健康检查
curl http://localhost:8080/health

# 批量预测接口
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      {
        "ctr": 0.15,
        "cvr": 0.08,
        "price": 99.0,
        "age": 25.0,
        "gender": 1.0,
        "age_x_ctr": 3.75,
        "gender_x_price": 99.0
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

与 Go 端 `RPCModel.Predict` 协议对齐：

```json
{
  "features_list": [
    {
      "ctr": 0.15,
      "cvr": 0.08,
      "price": 99.0,
      "age": 25.0,
      "gender": 1.0,
      "age_x_ctr": 3.75,
      "gender_x_price": 99.0
    }
  ]
}
```

**注意**: `gender` 值为 0=未知，1=男，2=女

### 响应格式

```json
{
  "scores": [0.85]
}
```

## 特征配置

特征列定义在 `train/features.py` 中：

- `ctr`: 点击率
- `cvr`: 转化率
- `price`: 价格
- `age`: 用户年龄
- `gender`: 用户性别（0=未知，1=男，2=女）
- `age_x_ctr`: 年龄 × CTR 交叉特征
- `gender_x_price`: 性别 × 价格交叉特征

## 扩展指南

### 使用自己的数据

1. 准备 CSV 数据文件，包含 `train/features.py` 中定义的特征列
2. 将数据文件放到 `data/` 目录
3. 修改 `train/train_xgb.py` 中的数据路径
4. 重新训练模型

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

### 构建镜像

```bash
docker build -t reckit-xgboost-service .
```

### 使用 Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 环境变量

- `PORT`: 服务端口（默认: 8080）
- `HOST`: 服务地址（默认: 0.0.0.0）
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

## 生产环境建议

- ✅ 使用 Docker 容器化部署（已实现）
- ✅ 添加模型版本管理（已实现）
- ✅ 实现特征标准化 Pipeline（已实现）
- ✅ 添加监控和日志（已实现）
- ⏳ 实现模型热更新（待实现）
- ⏳ 使用 gRPC 替代 HTTP（性能更好，待实现）
