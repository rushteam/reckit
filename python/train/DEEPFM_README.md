# DeepFM 模型训练与推理

本文档说明如何使用 PyTorch 训练 DeepFM 模型，并通过 HTTP 服务供 Golang 调用。

## DeepFM 模型简介

DeepFM 结合了 **Factorization Machine (FM)** 和 **Deep Neural Network (DNN)**：

- **FM 部分**：处理二阶特征交互（显式交互）
- **Deep 部分**：处理高阶非线性交互（隐式交互）
- **联合输出**：FM + Deep + Bias

**优势**：
- 自动学习特征交互，无需手动构造交叉特征
- 同时捕捉低阶和高阶特征交互
- 适合 CTR 预估、广告打分等场景

## 训练步骤

### 1. 安装依赖

```bash
cd python
pip install -r requirements.txt
# 或使用 uv
uv pip install -r requirements.txt
```

### 2. 准备训练数据

训练数据格式与 `train_xgb.py` 一致（CSV 格式，包含特征列和 `label` 列）：

```csv
item_ctr,item_cvr,item_price,user_age,user_gender,cross_age_x_ctr,cross_gender_x_price,label
0.15,0.08,99.0,25.0,1.0,3.75,99.0,1
0.12,0.05,150.0,30.0,2.0,3.6,300.0,0
...
```

如果数据文件不存在，训练脚本会自动生成示例数据。

### 3. 训练模型

```bash
cd python

# 基础训练（使用默认参数）
python train/train_deepfm.py

# 指定模型版本
python train/train_deepfm.py --version v1.0.0

# 自定义训练参数
python train/train_deepfm.py --epochs 100 --batch-size 64

# 组合使用
python train/train_deepfm.py --version v1.0.0 --epochs 100 --batch-size 64
```

**训练参数**：
- `--version`: 模型版本（可选，默认使用时间戳）
- `--epochs`: 训练轮数（默认 50）
- `--batch-size`: 批次大小（默认 32）

**输出文件**：
- `model/deepfm_model.pt`: PyTorch 模型文件
- `model/deepfm_feature_meta.json`: 特征元数据

### 4. 启动推理服务

```bash
cd python

# 使用 uvicorn
uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080

# 或直接运行
python service/deepfm_server.py

# 指定端口
PORT=8081 python service/deepfm_server.py
```

服务启动后，可通过以下接口调用：

- **健康检查**: `GET http://localhost:8080/health`
- **预测接口**: `POST http://localhost:8080/predict`

## Golang 端使用

在 Golang 中使用 `RPCNode` 调用 DeepFM 服务：

```go
import (
    "github.com/rushteam/reckit/model"
    "github.com/rushteam/reckit/rank"
    "time"
)

// 创建 RPC 模型
deepfmModel := model.NewRPCModel(
    "deepfm",
    "http://localhost:8080/predict", // DeepFM 服务端点
    5*time.Second,
)

// 在 Pipeline 中使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.RPCNode{
            Model: deepfmModel,
            // StripFeaturePrefix: false, // 默认不去掉前缀，与训练时特征名一致
        },
    },
}
```

## 请求/响应格式

### 请求格式（与 Go RPCModel 协议对齐）

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
    },
    {
      "item_ctr": 0.12,
      "item_cvr": 0.05,
      "item_price": 150.0,
      "user_age": 30.0,
      "user_gender": 2.0,
      "cross_age_x_ctr": 3.6,
      "cross_gender_x_price": 300.0
    }
  ]
}
```

### 响应格式

```json
{
  "scores": [0.85, 0.42]
}
```

## 模型配置

DeepFM 模型默认配置：

- **Embedding 维度**: 16
- **Deep 层结构**: [128, 64, 32]
- **Dropout**: 0.5
- **学习率**: 0.001
- **优化器**: Adam

可在 `train/train_deepfm.py` 中修改模型结构。

## 注意事项

1. **特征标准化**：DeepFM 训练时会自动进行特征标准化，推理时也会应用相同的标准化参数。
2. **特征顺序**：特征必须与训练时的 `FEATURE_COLUMNS` 顺序一致（通过 `feature_meta.json` 定义）。
3. **缺失特征**：如果请求中缺少某些特征，会使用默认值 0.0，并记录警告日志。
4. **GPU 支持**：如果系统有 CUDA，模型会自动使用 GPU 加速推理。

## 与 XGBoost 对比

| 维度 | XGBoost | DeepFM |
|------|---------|--------|
| **特征交互** | 树结构自动学习 | FM（二阶）+ DNN（高阶） |
| **训练速度** | 快 | 中等（需要 GPU 加速） |
| **推理速度** | 快 | 中等 |
| **可解释性** | 中等（特征重要性） | 弱（黑盒模型） |
| **适用场景** | 通用 | CTR 预估、广告打分 |

## 参考

- [排序模型文档](../../docs/RANK_MODELS.md)
- [模型选型指南](../../docs/MODEL_SELECTION.md)
- [RPC 召回示例](../../examples/rpc_xgb/main.go)
