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

接口定义在 `core` 包：

```go
import "github.com/rushteam/reckit/core"

type MLService interface {
    Predict(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error)
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
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/service"

// 使用 REST API
var tfService core.MLService = service.NewTFServingClient(
    "http://localhost:8501",
    "my_model",
    service.WithTFServingVersion("1"),
    service.WithTFServingTimeout(30*time.Second),
)

// 使用 gRPC（推荐）
tfService = service.NewTFServingClient(
    "localhost:8500",
    "my_model",
    service.WithTFServingGRPC(),
    service.WithTFServingVersion("1"),
)

// 批量预测
resp, err := tfService.Predict(ctx, &core.MLPredictRequest{
    Instances: [][]float64{
        {0.1, 0.2, 0.3, ...}, // 实例 1
        {0.4, 0.5, 0.6, ...}, // 实例 2
    },
})

// 健康检查
err := tfService.Health(ctx)
```

### 2. TorchServe

**协议**：
- REST API：端口 8080（推理）、8081（管理）
- gRPC API：端口 7070（需要额外配置）

**使用示例**：

```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/service"

// 使用 REST API
var torchService core.MLService = service.NewTorchServeClient(
    "http://localhost:8080",
    "my_model",
    service.WithTorchServeVersion("1.0"),
    service.WithTorchServeTimeout(30*time.Second),
)

// 批量预测
resp, err := torchService.Predict(ctx, &core.MLPredictRequest{
    Features: []map[string]float64{
        {"feature1": 0.1, "feature2": 0.2, ...}, // 实例 1
        {"feature1": 0.3, "feature2": 0.4, ...}, // 实例 2
    },
})

// 或使用 Instances（特征向量）
resp, err := torchService.Predict(ctx, &core.MLPredictRequest{
    Instances: [][]float64{
        {0.1, 0.2, 0.3, ...}, // 实例 1
        {0.4, 0.5, 0.6, ...}, // 实例 2
    },
})

// 健康检查
err := torchService.Health(ctx)
```

### 3. ANN Service

**协议**：HTTP/REST

**使用示例**：

```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/service"

// 注意：ANN（向量检索）应该使用 core.VectorService 接口，而不是 core.MLService
// 请参考 ext/vector/milvus 或 store.MemoryVectorService 实现向量检索
//
// 示例：
// import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// vectorService, _ := milvus.NewMilvusService("localhost:19530")
// result, err := vectorService.Search(ctx, &core.VectorSearchRequest{
//     Collection: "items",
//     Vector:     userVector,
//     TopK:       20,
//     Metric:     "cosine",
// })
```

## 工厂方法

使用工厂方法根据配置创建服务实例：

```go
import "github.com/rushteam/reckit/core"
import "github.com/rushteam/reckit/service"

// TF Serving 配置
config := &service.ServiceConfig{
    Type:        service.ServiceTypeTFServing,
    Endpoint:    "http://localhost:8501",
    ModelName:   "my_model",
    ModelVersion: "1",
    Timeout:     30,
}

// TorchServe 配置
torchConfig := &service.ServiceConfig{
    Type:        service.ServiceTypeTorchServe,
    Endpoint:    "http://localhost:8080",
    ModelName:   "my_model",
    ModelVersion: "1.0",
    Timeout:     30,
}

// 创建服务（返回 core.MLService）
mlService, err := service.NewMLService(config)
torchService, err := service.NewMLService(torchConfig)
if err != nil {
    // 处理错误
}
defer mlService.Close(ctx)

// 使用服务
resp, err := mlService.Predict(ctx, &core.MLPredictRequest{
    Instances: [][]float64{features},
})
```

## 请求/响应格式

### MLPredictRequest

类型定义在 `core` 包：

```go
import "github.com/rushteam/reckit/core"

type MLPredictRequest struct {
    Instances     [][]float64              // 特征向量列表
    Features      []map[string]float64     // 特征字典列表（可选）
    ModelName     string                   // 模型名称（可选）
    ModelVersion  string                   // 模型版本（可选）
    SignatureName string                   // 签名名称（可选）
    Params        map[string]interface{}   // 额外参数
}
```

### MLPredictResponse

类型定义在 `core` 包：

```go
type MLPredictResponse struct {
    Predictions  []float64    // 预测结果列表
    Outputs      interface{}  // 原始输出（可选）
    ModelVersion string       // 模型版本（如果服务返回）
}
```

## 与现有代码集成

### 1. 与 model.RankModel 集成

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/model"
    "github.com/rushteam/reckit/service"
)

// 创建 TF Serving 服务
var tfService core.MLService = service.NewTFServingClient("http://localhost:8501", "rank_model")

// 包装为 RankModel
rpcModel := model.NewRPCModelFromService("tf_serving", tfService)

// 在 Rank Node 中使用
rankNode := &rank.RPCNode{
    Model: rpcModel,
}
```

### 2. 与 core.VectorDatabaseService 集成

```go
import (
    "github.com/rushteam/reckit/service"
)

// 创建 ANN 服务客户端
// 注意：向量检索应使用 core.VectorService，请参考 ext/vector/milvus
// import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// vectorService, _ := milvus.NewMilvusService("localhost:19530")
//
// 在 recall.ANN 中使用
// ann := &recall.ANN{
//     VectorService: vectorService,  // 直接使用 core.VectorService
//     Collection:    "items",
//     TopK:          20,
// }
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

### TorchServe REST API

**请求格式**：
```json
POST /predictions/{model_name}
{
    "data": [{"feature1": 0.1, "feature2": 0.2, ...}, ...]
}
```

或使用特征向量：
```json
POST /predictions/{model_name}
{
    "data": [[0.1, 0.2, 0.3, ...], ...]
}
```

**响应格式**（取决于模型 Handler）：
```json
[0.85, 0.72, ...]
```

或：
```json
{
    "prediction": 0.85
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

### ✅ 已实现

- **TensorFlow Serving REST API**：完全支持
- **TorchServe REST API**：完全支持
- **ANN Service**：完全支持

### ⚠️ 部分实现

- **TensorFlow Serving gRPC**：接口已定义，需要 protobuf 依赖（当前回退到 REST API）

### 📝 待实现

- **TorchServe gRPC**：需要额外配置和 protobuf 依赖

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
- [TorchServe 文档](https://pytorch.org/serve/)
- [TorchServe REST API](https://pytorch.org/serve/rest_api.html)