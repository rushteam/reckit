# 特征元数据使用示例

本示例演示如何在 Go 代码中使用 `feature_meta.json` 和 `feature_scaler.json` 文件。

## 加载方式

支持三种加载方式，通过接口抽象实现：

### 方式 1：本地文件加载（开发环境）

```go
import (
    "context"
    "github.com/rushteam/reckit/feature"
)

ctx := context.Background()

// 使用文件加载器
fileMetaLoader := feature.NewFileMetadataLoader()
fileScalerLoader := feature.NewFileScalerLoader()

meta, err := fileMetaLoader.Load(ctx, "python/model/feature_meta.json")
if err != nil {
    log.Fatal(err)
}

scaler, err := fileScalerLoader.Load(ctx, "python/model/feature_scaler.json")
if err != nil {
    log.Fatal(err)
}
```

### 方式 2：HTTP 接口加载（生产环境）

```go
import (
    "context"
    "time"
    "github.com/rushteam/reckit/feature"
)

ctx := context.Background()

// 使用 HTTP 加载器
httpMetaLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
httpScalerLoader := feature.NewHTTPScalerLoader(5 * time.Second)

meta, err := httpMetaLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")
if err != nil {
    log.Fatal(err)
}

scaler, err := httpScalerLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_scaler")
if err != nil {
    log.Fatal(err)
}
```

### 方式 3：S3 兼容协议加载（推荐用于云环境）

**S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等**，使用统一的接口，兼容性更好。

```go
import (
    "context"
    "github.com/rushteam/reckit/feature"
)

// 需要实现 feature.S3Client 接口（S3 兼容协议）
type MyS3Client struct {
    // 实现 S3Client 接口
    // 可以使用 AWS S3 SDK，配置不同云服务商的 S3 兼容端点
}

func (c *MyS3Client) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
    // 实现 S3 兼容协议调用
    // 可以使用 github.com/aws/aws-sdk-go/service/s3
}

ctx := context.Background()
s3Client := &MyS3Client{}

// 使用 S3 兼容协议加载器
s3MetaLoader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
s3ScalerLoader := feature.NewS3ScalerLoader(s3Client, "my-bucket")

meta, err := s3MetaLoader.Load(ctx, "models/v1.0.0/feature_meta.json")
if err != nil {
    log.Fatal(err)
}

scaler, err := s3ScalerLoader.Load(ctx, "models/v1.0.0/feature_scaler.json")
if err != nil {
    log.Fatal(err)
}
```

**为什么使用 S3 兼容协议？**
- ✅ **统一接口**：一套代码支持多种对象存储服务
- ✅ **兼容性好**：AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等都支持 S3 兼容协议
- ✅ **易于迁移**：可以在不同云服务商之间轻松切换
- ✅ **生态丰富**：S3 兼容协议的 SDK 和工具更丰富

### 便捷函数（向后兼容）

```go
// 便捷函数，内部使用 FileMetadataLoader
meta, err := feature.LoadFeatureMetadata("python/model/feature_meta.json")
scaler, err := feature.LoadFeatureScaler("python/model/feature_scaler.json")
```

## 功能说明

### 1. 加载特征元数据

```go
fmt.Printf("模型版本: %s\n", meta.ModelVersion)
fmt.Printf("特征列: %v\n", meta.FeatureColumns)
fmt.Printf("是否标准化: %v\n", meta.Normalized)
```

### 2. 加载特征标准化器

```go
// 对特征进行标准化
normalized := scaler.Normalize(features)
```

### 3. 特征验证

```go
// 验证特征完整性，填充缺失值
validated := meta.ValidateFeatures(features)

// 检查缺失的特征
missing := meta.GetMissingFeatures(features)
if len(missing) > 0 {
    fmt.Printf("缺失特征: %v\n", missing)
}
```

### 4. 构建特征向量

```go
// 按 feature_columns 顺序构建特征向量
vector := meta.BuildFeatureVector(features)
// vector 是按 feature_columns 顺序的 []float64
```

### 5. 完整处理流程

```go
// 验证 + 标准化（如果配置了）
processed := meta.ProcessFeatures(features, scaler)
```

## 使用场景

### 场景 1：特征验证（推荐）

在发送特征到 Python 服务前，验证特征完整性：

```go
meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")

// 在 Pipeline 中添加验证节点
validationNode := &featureValidationNode{meta: meta}
pipeline.Nodes = append(pipeline.Nodes, validationNode)
```

### 场景 2：特征标准化（不推荐）

**注意**：当前架构中，标准化统一在 Python 服务中完成。如果需要在 Go 端做标准化，需要确保与 Python 服务的一致性。

```go
meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")
scaler, _ := feature.LoadFeatureScaler("python/model/feature_scaler.json")

if meta.Normalized {
    features = scaler.Normalize(features)
}
```

### 场景 3：特征监控

监控特征分布，检查 train-serving skew：

```go
meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")

// 检查缺失特征
missing := meta.GetMissingFeatures(features)
if len(missing) > 0 {
    log.Warnf("缺失特征: %v", missing)
}

// 检查特征值范围（需要额外的统计逻辑）
for _, col := range meta.FeatureColumns {
    if v, ok := features[col]; ok {
        // 检查是否在合理范围内
        // ...
    }
}
```

## 运行示例

```bash
# 1. 确保 Python 服务已启动
cd python
uvicorn service.server:app --host 0.0.0.0 --port 8080

# 2. 在另一个终端运行 Go 示例
cd examples/feature_metadata
go run main.go
```

## 注意事项

1. **文件路径**：确保 `feature_meta.json` 和 `feature_scaler.json` 的路径正确
2. **标准化位置**：当前架构标准化在 Python 服务中完成，Go 端通常只做验证
3. **特征名一致性**：确保 Go 端生成的特征名与 `feature_meta.json` 中的 `feature_columns` 完全一致
4. **可选加载**：如果文件不存在，可以跳过验证，让 Python 服务处理

## API 参考

### 加载器接口

#### MetadataLoader

- `Load(ctx context.Context, source string) (*FeatureMetadata, error)` - 加载特征元数据

**实现**：
- `FileMetadataLoader` - 本地文件加载器
- `HTTPMetadataLoader` - HTTP 接口加载器
- `OSSMetadataLoader` - OSS 文件加载器

#### ScalerLoader

- `Load(ctx context.Context, source string) (FeatureScaler, error)` - 加载特征标准化器

**实现**：
- `FileScalerLoader` - 本地文件加载器
- `HTTPScalerLoader` - HTTP 接口加载器
- `OSSScalerLoader` - OSS 文件加载器

### FeatureMetadata

- `LoadFeatureMetadata(path string) (*FeatureMetadata, error)` - 便捷函数，加载特征元数据（向后兼容）
- `ValidateFeatures(features map[string]float64) map[string]float64` - 验证特征，填充缺失值
- `GetMissingFeatures(features map[string]float64) []string` - 获取缺失的特征列
- `BuildFeatureVector(features map[string]float64) []float64` - 按顺序构建特征向量
- `ProcessFeatures(features map[string]float64, scaler FeatureScaler) map[string]float64` - 完整处理流程

### FeatureScaler

- `LoadFeatureScaler(path string) (FeatureScaler, error)` - 便捷函数，加载特征标准化器（向后兼容）
- `Normalize(features map[string]float64) map[string]float64` - 标准化特征
- `NormalizeValue(featureName string, value float64) float64` - 标准化单个特征值

### S3Client 接口

如果需要使用 S3 兼容协议加载器，需要实现 `S3Client` 接口：

```go
type S3Client interface {
    GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error)
}
```

**注意**：`NewOSSMetadataLoader` 和 `NewOSSScalerLoader` 已废弃，请使用 `NewS3MetadataLoader` 和 `NewS3ScalerLoader`。S3 兼容协议可以同时支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等。
