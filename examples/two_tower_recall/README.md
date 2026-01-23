# 双塔召回示例

本示例展示如何使用双塔模型进行召回，采用 **PyTorch 训练 + Golang 推理** 的架构。

## 架构设计

```
用户请求
    ↓
获取用户特征 (FeatureService)
    ↓
用户塔推理 (MLService: TorchServe/ONNX Runtime)
    ↓
User Embedding
    ↓
向量检索 (ANNService: Milvus/Faiss)
    ↓
Item IDs
```

## 核心组件

### 1. FeatureService（特征服务）
- **作用**：获取用户实时特征
- **实现**：支持 Memory、Redis、HTTP、Feast 等
- **优势**：Golang 高并发处理

### 2. MLService（模型推理服务）
- **作用**：运行用户塔推理，得到 User Embedding
- **实现**：支持 TorchServe、TensorFlow Serving、ONNX Runtime 等
- **模型**：PyTorch 训练的双塔模型（User Tower）

### 3. ANNService（向量检索服务）
- **作用**：根据 User Embedding 检索相似的 Item Embeddings
- **实现**：支持 Milvus、Faiss 等向量数据库
- **数据**：Item Embeddings 需要离线预计算并存入向量数据库

## 使用步骤

### 1. 准备模型服务

#### 启动 TorchServe（用户塔推理）

```bash
# 1. 将 PyTorch 模型转换为 TorchScript
torch-model-archiver \
  --model-name user_tower \
  --version 1.0 \
  --serialized-file user_tower.pth \
  --handler custom_handler.py \
  --export-path model_store

# 2. 启动 TorchServe
torchserve --start \
  --model-store model_store \
  --models user_tower=user_tower.mar \
  --ts-config config.properties
```

#### 启动 Milvus（向量检索）

```bash
# 使用 Docker Compose 启动 Milvus
docker-compose up -d
```

### 2. 准备数据

#### 离线预计算 Item Embeddings

```python
# Python 脚本：precompute_item_embeddings.py
import torch
import milvus

# 加载 Item Tower 模型
item_tower = torch.load('item_tower.pth')
item_tower.eval()

# 获取所有物品特征
items = get_all_items()

# 批量计算 Item Embeddings
item_embeddings = []
for item in items:
    features = extract_item_features(item)
    with torch.no_grad():
        embedding = item_tower(features)
        item_embeddings.append({
            'id': item.id,
            'vector': embedding.numpy().tolist()
        })

# 存入 Milvus
milvus_client.insert('item_embeddings', item_embeddings)
```

### 3. 运行示例

```bash
go run examples/two_tower_recall/main.go
```

## 配置说明

### TwoTowerRecall 配置

```go
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,      // 特征服务
    userTowerService,   // 用户塔推理服务
    vectorService,      // 向量检索服务
    recall.WithTwoTowerTopK(100),                    // 返回 Top 100
    recall.WithTwoTowerCollection("item_embeddings"), // 向量数据库集合名称
    recall.WithTwoTowerMetric("inner_product"),      // 使用内积（适合双塔模型）
)
```

### 特征服务配置

```go
// 方式1：内存特征服务（测试用）
featureService := feature.NewMemoryFeatureService()

// 方式2：HTTP 特征服务
featureService := feature.NewHTTPFeatureService(
    "http://localhost:8080/features",
    5*time.Second,
)

// 方式3：Redis 特征服务
featureService := feature.NewRedisFeatureService(
    redisClient,
    "feature:",
)
```

### 模型服务配置

```go
// TorchServe
userTowerService := service.NewTorchServeClient(
    "http://localhost:8080",
    "user_tower",
    service.WithTorchServeTimeout(5*time.Second),
)

// TensorFlow Serving
userTowerService := service.NewTFServingClient(
    "localhost:8500",
    "user_tower",
    "v1",
)
```

### 向量服务配置

```go
// Milvus
vectorService := vector.NewMilvusService(
    "localhost:19530",
    vector.WithMilvusDatabase("default"),
    vector.WithMilvusTimeout(10),
)
```

## 性能优化

### 1. 特征服务优化
- 使用批量获取：`BatchGetUserFeatures`
- 使用缓存：Redis 缓存用户特征
- 异步获取：非关键特征异步获取

### 2. 模型服务优化
- 批量推理：一次推理多个用户（如果支持）
- 连接池：复用 HTTP/gRPC 连接
- 超时控制：设置合理的超时时间

### 3. 向量检索优化
- 索引优化：使用 HNSW、IVF 等索引
- 批量检索：一次检索多个向量（如果支持）
- 缓存结果：缓存热门用户的召回结果

## 注意事项

### 1. 特征一致性
- **训练和推理特征必须一致**：确保特征名称、类型、顺序一致
- **推荐使用字典格式**：使用 `Features`（字典格式）而不是 `Instances`（向量格式），避免特征顺序问题

### 2. Embedding 维度
- **User Embedding 和 Item Embedding 维度必须一致**
- **常见维度**：32、64、128、256

### 3. 距离度量
- **内积（inner_product）**：适合双塔模型，计算速度快
- **余弦相似度（cosine）**：归一化后计算，对向量长度不敏感
- **欧氏距离（euclidean）**：不常用，计算较慢

### 4. 模型部署
- **Item Embedding 需要离线预计算**：定期更新（例如：每天）
- **User Embedding 实时计算**：每次请求实时计算（用户特征变化频繁）

## 扩展性

### 替换实现

所有组件都通过接口实现，可以轻松替换：

```go
// 替换特征服务
featureService := feature.NewFeastFeatureService(...)

// 替换模型服务
userTowerService := service.NewONNXRuntimeClient(...)

// 替换向量服务
vectorService := vector.NewFaissService(...)
```

### 自定义特征提取器

```go
twoTowerRecall := recall.NewTwoTowerRecall(
    featureService,
    userTowerService,
    vectorService,
    recall.WithTwoTowerUserFeatureExtractor(func(ctx context.Context, rctx *core.RecommendContext) (map[string]float64, error) {
        // 自定义特征提取逻辑
        features := make(map[string]float64)
        // ...
        return features, nil
    }),
)
```

## 相关文档

- [双塔模型搭建指南](../../docs/TWO_TOWER_GUIDE.md)
- [模型选型指南](../../docs/MODEL_SELECTION.md)
- [召回算法文档](../../docs/RECALL_ALGORITHMS.md)
