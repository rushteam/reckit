# ML Service 标准接口

Reckit 与 TF Serving / ANN 服务的标准接口设计，提供统一的机器学习服务对接能力。

## 设计目标

- **统一接口**：提供统一的 MLService 接口，支持 TF Serving、TorchServe、ANN 等服务
- **协议支持**：支持 gRPC（TF Serving）和 HTTP/REST
- **批量预测**：支持批量请求，提高性能
- **错误处理**：统一的错误处理和重试机制
- **配置化**：通过配置创建服务实例

## 核心接口

### MLService

```go
type MLService interface {
    Predict(ctx context.Context, req *PredictRequest) (*PredictResponse, error)
    Health(ctx context.Context) error
    Close() error
}
```

## 服务类型

### 1. TensorFlow Serving

**协议**：
- gRPC：端口 8500（推荐，性能更好）
- REST API：端口 8501（HTTP/JSON）

**使用示例**：

```go
import "github.com/rushteam/reckit/service"

// 使用 REST API
tfService := service.NewTFServingClient(
    "http://localhost:8501",
    "my_model",
    service.WithTFServingVersion("1"),
    service.WithTFServingTimeout(30*time.Second),
)

// 使用 gRPC（推荐）
tfService := service.NewTFServingClient(
    "localhost:8500",
    "my_model",
    service.WithTFServingGRPC(),
    service.WithTFServingVersion("1"),
)

// 批量预测
resp, err := tfService.Predict(ctx, &service.PredictRequest{
    Instances: [][]float64{
        {0.1, 0.2, 0.3, ...}, // 实例 1
        {0.4, 0.5, 0.6, ...}, // 实例 2
    },
})

// 健康检查
err := tfService.Health(ctx)
```

### 2. ANN Service

**协议**：HTTP/REST

**使用示例**：

```go
import "github.com/rushteam/reckit/service"

// 创建 ANN 服务客户端
annService := service.NewANNServiceClient(
    "http://localhost:19530",
    "items",
    service.WithANNServiceTimeout(30*time.Second),
)

// 向量搜索
ids, scores, err := annService.Search(ctx, userVector, 20, "cosine")

// 或使用 Predict 接口
resp, err := annService.Predict(ctx, &service.PredictRequest{
    Instances: [][]float64{userVector},
    Params: map[string]interface{}{
        "top_k":  20,
        "metric": "cosine",
    },
})
```

## 工厂方法

使用工厂方法根据配置创建服务实例：

```go
import "github.com/rushteam/reckit/service"

// TF Serving 配置
config := &service.ServiceConfig{
    Type:        service.ServiceTypeTFServing,
    Endpoint:    "http://localhost:8501",
    ModelName:   "my_model",
    ModelVersion: "1",
    Timeout:     30,
}

// 创建服务
mlService, err := service.NewMLService(config)
if err != nil {
    // 处理错误
}
defer mlService.Close()

// 使用服务
resp, err := mlService.Predict(ctx, &service.PredictRequest{
    Instances: [][]float64{features},
})
```

## 请求/响应格式

### PredictRequest

```go
type PredictRequest struct {
    Instances     [][]float64              // 特征向量列表
    Features      []map[string]float64     // 特征字典列表（可选）
    ModelName     string                   // 模型名称（可选）
    ModelVersion  string                   // 模型版本（可选）
    SignatureName string                   // 签名名称（可选）
    Params        map[string]interface{}   // 额外参数
}
```

### PredictResponse

```go
type PredictResponse struct {
    Predictions  []float64    // 预测结果列表
    Outputs      interface{}  // 原始输出（可选）
    ModelVersion string       // 模型版本（如果服务返回）
}
```

## 与现有代码集成

### 1. 与 model.RankModel 集成

```go
import (
    "github.com/rushteam/reckit/model"
    "github.com/rushteam/reckit/service"
)

// 创建 TF Serving 服务
tfService := service.NewTFServingClient("http://localhost:8501", "rank_model")

// 包装为 RankModel
rpcModel := model.NewRPCModelFromService("tf_serving", tfService)

// 在 Rank Node 中使用
rankNode := &rank.RPCNode{
    Model: rpcModel,
}
```

### 2. 与 vector.ANNService 集成

```go
import (
    "github.com/rushteam/reckit/vector"
    "github.com/rushteam/reckit/service"
)

// 创建 ANN 服务客户端
annClient := service.NewANNServiceClient("http://localhost:19530", "items")

// 包装为 vector.ANNService
annService := vector.NewANNServiceFromClient(annClient)

// 在 recall.ANN 中使用
ann := &recall.ANN{
    Store: vector.NewVectorStoreAdapter(annService, "items"),
    TopK:  20,
}
```

## 协议规范

### TF Serving REST API

**请求格式**：
```json
POST /v1/models/{model_name}:predict
{
    "instances": [[f1, f2, f3, ...], ...],
    "signature_name": "serving_default"
}
```

**响应格式**：
```json
{
    "predictions": [score1, score2, ...]
}
```

### ANN Service HTTP API

**请求格式**：
```json
POST /v1/vector/search
{
    "collection": "items",
    "vectors": [[v1, v2, v3, ...], ...],
    "top_k": 10,
    "metric": "cosine"
}
```

**响应格式**：
```json
{
    "results": [
        {
            "ids": [1, 2, 3, ...],
            "scores": [0.95, 0.92, 0.88, ...]
        }
    ]
}
```

## 实现状态

当前为**占位实现**，接口已定义但未完整实现。实际实现需要：

### TF Serving

1. **安装依赖**：
   ```bash
   go get google.golang.org/grpc
   go get github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf
   ```

2. **实现 gRPC 客户端**：
   - 连接管理
   - 请求构建
   - 响应解析

3. **实现 REST 客户端**：
   - HTTP 请求
   - JSON 序列化/反序列化

### ANN Service

1. **实现 HTTP 客户端**：
   - 请求构建
   - 响应解析
   - 错误处理

## 使用示例

完整示例代码请参考：`examples/ml_service/main.go`

运行示例：
```bash
go run ./examples/ml_service
```

## 参考

- [TensorFlow Serving 文档](https://www.tensorflow.org/tfx/guide/serving)
- [TensorFlow Serving REST API](https://www.tensorflow.org/tfx/serving/api_rest)
- [TensorFlow Serving gRPC API](https://www.tensorflow.org/tfx/serving/api_rest)
