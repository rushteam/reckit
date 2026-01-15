# Embedding 能力抽象

## 概述

项目提供了基于 Embedding 向量的近似最近邻（ANN）召回能力，支持通过向量相似度计算实现个性化物品召回。

## 核心组件

### 1. ANN 召回器 (`recall.ANN`)

**位置**: `recall/ann.go`

**功能**:
- 基于用户向量和物品向量的相似度计算进行召回
- 支持多种相似度度量方式（余弦相似度、欧氏距离）
- 支持 TopK 检索
- 可集成到 Pipeline 中作为召回源

**核心结构**:
```go
type ANN struct {
    Store      VectorStore  // 向量存储接口
    Key        string       // 向量索引 key（预留）
    UserVector []float64    // 用户向量（可选，优先使用）
    TopK       int          // 返回 TopK 相似物品
    Metric     string       // 距离度量：cosine / euclidean
    
    // 用户向量提取器（可选）
    UserVectorExtractor func(rctx *core.RecommendContext) []float64
}
```

### 2. VectorStore 接口

**位置**: `recall/ann.go`

**定义**:
```go
type VectorStore interface {
    GetVector(ctx context.Context, itemID int64) ([]float64, error)
    ListVectors(ctx context.Context) (map[int64][]float64, error)
}
```

**职责**:
- 提供物品向量的存储和检索能力
- 支持单个向量查询和批量向量查询
- 可扩展实现（Redis、内存、向量数据库如 Faiss、Milvus 等）

## 用户向量获取策略

支持三种方式获取用户向量，优先级从高到低：

1. **直接提供** (`UserVector`): 在创建 ANN 时直接指定用户向量
2. **自定义提取器** (`UserVectorExtractor`): 通过函数从 `RecommendContext` 中提取
3. **默认提取** (`rctx.UserProfile["user_vector"]`): 从上下文的用户画像中获取

**示例**:
```go
// 方式1: 直接提供
ann := &recall.ANN{
    UserVector: []float64{0.1, 0.2, 0.3},
    // ...
}

// 方式2: 自定义提取器
ann := &recall.ANN{
    UserVectorExtractor: func(rctx *core.RecommendContext) []float64 {
        // 自定义提取逻辑
        return extractUserVector(rctx)
    },
    // ...
}

// 方式3: 从 Context 获取
rctx := &core.RecommendContext{
    UserProfile: map[string]any{
        "user_vector": []float64{0.1, 0.2, 0.3},
    },
}
```

## 相似度计算

### 支持的度量方式

1. **余弦相似度** (`cosine`): 默认方式
   - 计算公式: `cos(θ) = (A·B) / (||A|| * ||B||)`
   - 范围: [-1, 1]，值越大越相似
   - 适用于方向相似性判断

2. **欧氏距离** (`euclidean`): 
   - 计算公式: `distance = sqrt(Σ(Ai - Bi)²)`
   - 转换为相似度: `similarity = 1 / (1 + distance)`
   - 适用于绝对距离判断

### 实现细节

- 余弦相似度: `cosineSimilarity()` 函数
- 欧氏距离: `euclideanDistance()` 函数
- TopK 选择: 使用选择排序（生产环境建议使用堆优化）

## 集成方式

### 作为召回源使用

ANN 实现了 `recall.Source` 接口，可以：

1. **单独使用**:
```go
ann := &recall.ANN{
    Store:      vectorStore,
    TopK:       20,
    Metric:     "cosine",
}
items, err := ann.Recall(ctx, rctx)
```

2. **集成到 Fanout**（多路召回）:
```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{...},
        &recall.ANN{...},  // Embedding 召回
        &recall.UserHistory{...},
    },
    MergeStrategy: "priority",
}
```

3. **集成到 Pipeline**:
```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.ANN{...},
            },
        },
        // 其他节点...
    },
}
```

## 输出结果

召回结果包含以下信息：

- **Item ID**: 物品标识
- **Score**: 相似度分数
- **Labels**:
  - `recall_source`: "ann"
  - `ann_metric`: 使用的度量方式（"cosine" 或 "euclidean"）

## 扩展点

### 1. VectorStore 实现

可以扩展实现不同的向量存储后端：

- **内存实现**: 适用于小规模数据或测试
- **Redis 实现**: 利用 Redis 的 Hash 或 String 存储向量
- **专业向量数据库**: 集成 Faiss、Milvus、Qdrant 等，支持高效 ANN 检索

### 2. 优化方向

当前实现为简化版本，生产环境可优化：

- **索引优化**: 使用向量索引（如 HNSW、IVF）替代全量扫描
- **TopK 优化**: 使用堆排序替代选择排序
- **并发优化**: 支持向量检索的并发处理
- **缓存优化**: 对常用向量进行缓存

## 使用场景

1. **个性化推荐**: 基于用户 Embedding 召回相似物品
2. **内容相似性**: 基于物品 Embedding 进行相似物品推荐
3. **冷启动**: 新用户/新物品的向量化召回
4. **多模态推荐**: 支持文本、图像等 Embedding 的召回

## 注意事项

1. **向量维度**: 确保用户向量和物品向量维度一致
2. **向量归一化**: 建议对向量进行归一化处理，特别是使用余弦相似度时
3. **性能考虑**: 当前实现为全量扫描，大规模数据建议使用专业向量数据库
4. **向量更新**: 需要建立向量更新机制，保证向量数据的时效性
